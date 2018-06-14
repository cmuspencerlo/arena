import cv2
import sys
sys.path.append("game/")
import wrapped_flappy_bird as game
import random
import numpy as np
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from tensorboardX import SummaryWriter

GAME = 'bird' # the name of the game being played for log files
NUM_ACTION = 2 # number of valid actions
MEMORY_LEN = 5 # number of previous transitions to remember
BATCH = 3 # size of minibatch
TRAINING_EPOCH = 10
EPSILON = 1e-6
GAMMA = 0.99 # decay rate of past observations
NUM_STACK = 6

OBSERVE = 100000. # timesteps to observe before training
EXPLORE = 2000000. # frames over which to anneal epsilon
FINAL_EPSILON = 0.0001 # final value of epsilon
INITIAL_EPSILON = 0.0001 # starting value of epsilon

class FlappyQNet(nn.Module):
    def __init__(self):
        super(FlappyQNet, self).__init__()
        self.conv1 = nn.Conv2d(NUM_STACK, 32, 8, 4, 2)
        self.conv2 = nn.Conv2d(32, 64, 4, 2, 2)
        self.conv3 = nn.Conv2d(64, 64, 3, 1, 2)
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, NUM_ACTION)

    def forward(self, s):
        x = self.conv1(s)
        x = F.max_pool2d(F.relu(x), 2)

        x = self.conv2(x)
        x = F.max_pool2d(F.relu(x), 2)

        x = self.conv3(x)
        x = F.max_pool2d(F.relu(x), 2)

        x = x.view(-1, 256)
        x = F.relu(self.fc1(x))
        out = self.fc2(x)

        return out

def random_s_init(game_state):
    s_t = None
    for _ in range(3):
        a_t = np.zeros(NUM_ACTION)
        a_t[random.randrange(NUM_ACTION)] = 1
        _, s_t, _ = action_stack_state(game_state, a_t, s_t, init_flag=True)
    return s_t

def action_stack_state(game_state, a_t, s_t, init_flag=False):
    x_t1_colored, r_t, terminal = game_state.frame_step(a_t)
    x_t1 = cv2.cvtColor(cv2.resize(x_t1_colored, (80, 80)), cv2.COLOR_BGR2GRAY)
    _, x_t1 = cv2.threshold(x_t1, 1, 255, cv2.THRESH_BINARY)
    x_t1 = np.reshape(x_t1, (1, 80, 80))
    if init_flag:
        s_t1 = np.stack(([x_t1]*NUM_STACK), axis=0).squeeze(axis=1)
        s_t1 = np.expand_dims(s_t1, 0)
    else:
        s_t = s_t.squeeze(axis=0)
        s_t1 = np.append(x_t1, s_t[:NUM_STACK - 1, :, :], axis=0)
        s_t1 = np.expand_dims(s_t1, 0)
    return r_t, s_t1, terminal

if __name__ == "__main__":
    game_state = game.GameState()
    experience = deque(maxlen=MEMORY_LEN)
    terminal = False

    # Fill the experience
    # input_actions[0] == 1: do nothing
    # input_actions[1] == 1: flap the bird
    while len(experience) < MEMORY_LEN:
        s_t = random_s_init(game_state)
        while terminal == False:
            a_t = np.zeros(NUM_ACTION)
            a_t[random.randint(0, 1)] = 1
            r_t, s_t1, terminal = action_stack_state(game_state, a_t, s_t)
            # (80, 80, 6)
            m_s_t = torch.from_numpy(s_t).float().cuda()
            m_a_t = torch.from_numpy(a_t).float().cuda()
            # m_r_t = torch.from_numpy(np.array(r_t)).float().cuda()
            m_s_t1 = torch.from_numpy(s_t1).float().cuda()
            experience.append((m_s_t, m_a_t, r_t, m_s_t1, terminal))
            s_t = s_t1

        criterion = nn.MSELoss()
        net = FlappyQNet()
        net = net.cuda()
        optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
        # Training process:
        for i in range(TRAINING_EPOCH):
            s_t = random_s_init(game_state)
            terminal = False
            while terminal == False:
                a_t = np.zeros(NUM_ACTION)
                # epsilon-greedy exploration
                if random.random() < EPSILON:
                    print("----------Random Action----------")
                    a_t[random.randrange(NUM_ACTION)] = 1
                else:
                    net.eval()
                    m_s_t = torch.from_numpy(s_t).float().cuda()
                    q_t = net(Variable(m_s_t, volatile=True))
                    # draw action fro q_t
                    a_t[torch.max(q_t, dim=1)[1].data] = 1

                r_t, s_t1, terminal = action_stack_state(game_state, a_t, s_t)
                m_a_t = torch.from_numpy(a_t).float().cuda()
                # m_r_t = torch.from_numpy(np.array(r_t)).float().cuda()
                m_s_t1 = torch.from_numpy(s_t1).float().cuda()
                experience.append((m_s_t, m_a_t, r_t, m_s_t1, terminal))

                # random draw from queue
                minibatch = random.sample(experience, BATCH)
                s_t_batch = [d[0] for d in minibatch]
                s_t_batch = torch.stack(s_t_batch, dim=1).squeeze(0)
                a_t_batch = [d[1] for d in minibatch]
                a_t_batch = torch.stack(a_t_batch, dim=1).squeeze(0).transpose(0, 1)
                r_t_batch = [d[2] for d in minibatch]
                # r_t_batch = torch.stack(r_t_batch, dim=1).squeeze(0)
                s_t1_batch = [d[3] for d in minibatch]
                s_t1_batch = torch.stack(s_t1_batch, dim=1).squeeze(0)

                # calculate q_target
                net.eval()
                target_q_batch = list()
                q_t1_batch = net(Variable(s_t1_batch, volatile=True))
                for index in range(len(minibatch)):
                    terminal = minibatch[index][4]
                    if terminal:
                        target_q_batch.append(Variable(torch.Tensor([r_t_batch[index]]).cuda()))
                    else:
                        target_q_batch.append(r_t_batch[index] + GAMMA * torch.max(q_t1_batch[index]))

                net.train()
                q_t_batch = net(Variable(s_t_batch))
                q_t_batch = torch.sum(q_t_batch * Variable(a_t_batch), dim=1)
                print(q_t_batch)
                target_q_batch = torch.stack(target_q_batch, dim=0)
                print(target_q_batch)
                optimizer.zero_grad()
                loss = criterion(q_t_batch, target_q_batch)
                loss.backward()
                optimizer.step()


    #         evaluate_network
    #         draw tab



