# -*- coding: utf-8 -*-
from io import open
import unicodedata
import string
import re
import random

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
from tensorboardX import SummaryWriter
import uuid

writer = SummaryWriter('logs/%s' % uuid.uuid1())

SOS_token = 0
EOS_token = 1

class Language:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS

    def add_sentence(self, sentence):
        for word in sentence.split(' '):
            self.add_word(word)

    def add_word(self, word):
        if word in self.word2index:
            self.word2count[word] += 1
        else:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1

def unicode_ascii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s)
                   if unicodedata.category(c) != 'Mn')

def preprocess_string(s):
    s = unicode_ascii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s

def read_language(source, destination):
    print('Reading...')
    string_pairs = []
    with open('data/%s-%s.txt' % (source, destination), encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            string_pairs.append([preprocess_string(s) for s in line.split('\t')])
    input_lang = Language(source)
    output_lang = Language(destination)

    return input_lang, output_lang, string_pairs

MAX_LENGTH = 25

eng_prefixes = (
    "i am ", "i m ",
    "he is", "he s ",
    "she is", "she s",
    "you are", "you re ",
    "we are", "we re ",
    "they are", "they re "
)

def filter_pair(p):
    return len(p[0].split(' ')) < MAX_LENGTH and \
           len(p[1].split(' ')) < MAX_LENGTH and \
           p[0].startswith(eng_prefixes)

def preprocess(pairs):
    return [p for p in pairs if filter_pair(p)]

def read_data(source, destination):
    source_lang, destination_lang, pairs = read_language(source, destination)
    pairs = preprocess(pairs)
    print("Read %s valid sentence pairs" % len(pairs))
    for p in pairs:
        source_lang.add_sentence(p[0])
        destination_lang.add_sentence(p[1])
    print("Counted words:")
    print(source_lang.name, source_lang.n_words)
    print(destination_lang.name, destination_lang.n_words)
    return source_lang, destination_lang, pairs

input_lang, output_lang, pairs = read_data('eng', 'fra')
print(random.choice(pairs))

######################################################################
# The Seq2Seq Model
# =================
#
# A Recurrent Neural Network, or RNN, is a network that operates on a
# sequence and uses its own output as input for subsequent steps.
#
# With a seq2seq model the encoder creates a single vector which, in the
# ideal case, encodes the "meaning" of the input sequence into a single
# vector — a single point in some N dimensional space of sentences.
#

######################################################################
# The Encoder
# -----------
#
# The encoder of a seq2seq network is a RNN that outputs some value for
# every word from the input sentence. For every input word the encoder
# outputs a vector and a hidden state, and uses the hidden state for the
# next input word.
#

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        # input_size = word_size
        # different from input below
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.hidden_size = hidden_size
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)

    def forward(self, input, hidden):
        # embedding leads to (1, size)
        input = self.embedding(input).view(1, 1, -1)
        # input: shape(batch, seq_size, input_size):
        output, hidden = self.gru(input, hidden)
        return output, hidden

    def init_hidden(self):
        return Variable(torch.ones(1, 1, self.hidden_size)).cuda()
#
#
# ######################################################################
# # Attention Decoder
# # ^^^^^^^^^^^^^^^^^
# #
# # If only the context vector is passed betweeen the encoder and decoder,
# # that single vector carries the burden of encoding the entire sentence.
# #
# # Attention allows the decoder network to "focus" on a different part of
# # the encoder's outputs for every step of the decoder's own outputs. First
# # we calculate a set of *attention weights*. These will be multiplied by
# # the encoder output vectors to create a weighted combination. The result
# # (called ``attn_applied`` in the code) should contain information about
# # that specific part of the input sequence, and thus help the decoder
# # choose the right output words.
# #
# #
# # Calculating the attention weights is done with another feed-forward
# # layer ``attn``, using the decoder's input and hidden state as inputs.
# # Because there are sentences of all sizes in the training data, to
# # actually create and train this layer we have to choose a maximum
# # sentence length (input length, for encoder outputs) that it can apply
# # to. Sentences of the maximum length will use all the attention weights,
# # while shorter sentences will only use the first few.

class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout=0.1):
        super(AttnDecoderRNN, self).__init__()
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.attn = nn.Linear(hidden_size * 2, MAX_LENGTH)
        self.attn_combine = nn.Linear(hidden_size * 2, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedding = self.dropout(self.embedding(input).view(1, 1, -1))
        # embedding: (batch, step, hidden)
        attn_weights = F.softmax(self.attn(torch.cat((embedding[0], hidden[0]), dim=1)), dim=1)
        attn_applied = torch.matmul(attn_weights, encoder_outputs)
    #         attn_applied = torch.bmm(attn_weights.unsqueeze(0),
    #                                  encoder_outputs.unsqueeze(0))
        input = self.attn_combine(torch.cat((embedding[0], attn_applied), dim=1))
        input = F.relu(input).view(1, 1, -1)
        output, hidden = self.gru(input, hidden)
        # output = F.log_softmax(self.out(output[0]), dim=1)
        output = self.out(output[0])
        return output, hidden, attn_weights

    def init_hidden(self):
        return Variable(torch.ones(1, 1, self.hidden_size)).cuda()

######################################################################
# .. note:: There are other forms of attention that work around the length
#   limitation by using a relative position approach. Read about "local
#   attention" in `Effective Approaches to Attention-based Neural Machine
#   Translation <https://arxiv.org/abs/1508.04025>`__.

def sentence_to_var(language, sentence):
    indices = [language.word2index[word] for word in sentence.split(' ')]
    indices.append(EOS_token)
    return Variable(torch.LongTensor(indices)).cuda()
#     result = Variable(torch.LongTensor(indexes).view(-1, 1))

def pair_to_var(pair):
    input_variable = sentence_to_var(input_lang, pair[0])
    target_variable = sentence_to_var(output_lang, pair[1])
    return [input_variable, target_variable]

def train(pairs, encoder, decoder, n_iters, learning_rate=0.01):
    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    # encoder_optimizer = optim.Adam(encoder.parameters())
    # decoder_optimizer = optim.Adam(decoder.parameters())
    criterion = nn.CrossEntropyLoss()
    training_pairs = [pair_to_var(random.choice(pairs)) for i in range(n_iters)]

    for i in range(n_iters):
        training_pair = training_pairs[i]
        input_var, target_var = training_pair[0], training_pair[1]

        encoder_hidden = encoder.init_hidden()

        input_length = input_var.size()[0]
        target_length = target_var.size()[0]

        encoder_outputs = Variable(torch.zeros(MAX_LENGTH, encoder.hidden_size)).cuda()

        # input_var: ['32', '11', '4'] => English
        for index in range(input_length):
            encoder_output, encoder_hidden = encoder(input_var[index], encoder_hidden)
            encoder_outputs[index] = encoder_output

        decoder_input = Variable(torch.LongTensor([SOS_token])).cuda()
        decoder_hidden = encoder_hidden

        loss = 0
        teacher_flag = True if random.random() < 0.8 else False
        for index in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output, target_var[index])
            if teacher_flag:
                decoder_input = target_var[index]
            else:
                value, topi = decoder_output.data.topk(1)
                if topi[0][0] == EOS_token:
                    break
                else:
                    decoder_input = Variable(torch.LongTensor([topi[0][0]])).cuda()
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()
        loss.backward()
        encoder_optimizer.step()
        decoder_optimizer.step()

        if i % 200 == 0:
            print('%d: loss:%.4f' % (i, loss.data[0] / target_length))

        writer.add_scalar('loss', loss.data[0] / target_length, i)

def evaluate(pairs, encoder, decoder, cnt=10):
    encoder.eval()
    decoder.eval()
    evaluate_pairs = [pair_to_var(random.choice(pairs)) for i in range(cnt)]
    for i in range(cnt):
        input_var, groundtruth_var = evaluate_pairs[i][0], evaluate_pairs[i][1]
        encoder_hidden = encoder.init_hidden()
        input_length = input_var.size()[0]
        encoder_outputs = Variable(torch.zeros(MAX_LENGTH, encoder.hidden_size)).cuda()
        for index in range(input_length):
            encoder_output, encoder_hidden = encoder(input_var[index], encoder_hidden)
            encoder_outputs[index] = encoder_output

        decoder_input = Variable(torch.LongTensor([SOS_token])).cuda()
        decoder_hidden = encoder_hidden
        decoder_output_list = []
        for index in range(MAX_LENGTH):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            _, topi = decoder_output.data.topk(1)
            if topi[0][0] == EOS_token:
                break
            else:
                decoder_input = Variable(torch.LongTensor([topi[0][0]])).cuda()
                decoder_output_list.append(output_lang.index2word[topi[0][0]])

        print(translate_words(input_lang, input_var.data.cpu()))
        print(translate_words(output_lang, groundtruth_var.data.cpu()))
        print('>> {}'.format(decoder_output_list))

def translate_words(lang, tensors):
    return ' '.join(lang.index2word[t] for t in tensors)

#
# def evaluateRandomly(encoder, decoder, n=10):
#     for i in range(n):
#         pair = random.choice(pairs)
#         print('>', pair[0])
#         print('=', pair[1])
#         output_words, attentions = evaluate(encoder, decoder, pair[0])
#         output_sentence = ' '.join(output_words)
#         print('<', output_sentence)
#         print('')
#
#
hidden_size = 1024
encoder = EncoderRNN(input_lang.n_words, hidden_size).cuda()
attention_decoder = AttnDecoderRNN(hidden_size, output_lang.n_words, dropout=0.1).cuda()

train(pairs, encoder, attention_decoder, 1000)
evaluate(pairs, encoder, attention_decoder)

writer.close()

# ######################################################################
# #
#
# evaluateRandomly(encoder1, attn_decoder1)
#
# ######################################################################
# # Visualizing Attention
# # ---------------------
# #
# # A useful property of the attention mechanism is its highly interpretable
# # outputs. Because it is used to weight specific encoder outputs of the
# # input sequence, we can imagine looking where the network is focused most
# # at each time step.
# #
# # You could simply run ``plt.matshow(attentions)`` to see attention output
# # displayed as a matrix, with the columns being input steps and rows being
# # output steps:
# #
#
#
# output_words, attentions = evaluate(
#     encoder1, attn_decoder1, "je suis trop froid .")
# plt.matshow(attentions.numpy())
#
#
# def showAttention(input_sentence, output_words, attentions):
#     # 用颜色条设置图形
#     fig = plt.figure()
#     ax = fig.add_subplot(111)
#     cax = ax.matshow(attentions.numpy(), cmap='bone')
#     fig.colorbar(cax)
#
#     # 设置轴
#     ax.set_xticklabels([''] + input_sentence.split(' ') +
#                        ['<EOS>'], rotation=90)
#     ax.set_yticklabels([''] + output_words)
#
#     # 在每个打勾处显示标签
#     ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
#     ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
#
#     plt.show()
#
#
# def evaluateAndShowAttention(input_sentence):
#     output_words, attentions = evaluate(
#         encoder1, attn_decoder1, input_sentence)
#     print('input =', input_sentence)
#     print('output =', ' '.join(output_words))
#     showAttention(input_sentence, output_words, attentions)
#
#
# evaluateAndShowAttention("elle a cinq ans de moins que moi .")
#
# evaluateAndShowAttention("elle est trop petit .")
#
# evaluateAndShowAttention("je ne crains pas de mourir .")
#
# evaluateAndShowAttention("c est un jeune directeur plein de talent .")


######################################################################
# Exercises
# =========
#
# -  Try with a different dataset
#
#    -  Another language pair
#    -  Human → Machine (e.g. IOT commands)
#    -  Chat → Response
#    -  Question → Answer
#
# -  Replace the embeddings with pre-trained word embeddings such as word2vec or
#    GloVe
# -  Try with more layers, more hidden units, and more sentences. Compare
#    the training time and results.
# -  If you use a translation file where pairs have two of the same phrase
#    (``I am test \t I am test``), you can use this as an autoencoder. Try
#    this:
#
#    -  Train as an autoencoder
#    -  Save only the Encoder network
#    -  Train a new Decoder for translation from there
#

