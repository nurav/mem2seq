import torch
import numpy as np

from torch import nn
from masked_cross_entropy import*

import numpy as np
import logging
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torch import optim
import torch.nn.functional as F
from masked_cross_entropy import *

USE_CUDA = torch.cuda.is_available()

TYPE = torch.LongTensor
TYPEF = torch.FloatTensor
use_cuda = torch.cuda.is_available()

if use_cuda:
    TYPE = torch.cuda.LongTensor
    TYPEF = torch.cuda.FloatTensor
    # model.cuda()

class Model(nn.Module):
    # def __init__(self, gru_size, nwords, lr, hops, dropout, unk_mask):
    def __init__(self, hops, nwords, emb_size, gru_size, w2i):

        super(Model, self).__init__()
        self.name = "Mem2Seq"
        self.nwords = nwords
        self.gru_size = gru_size
        self.emb_size = emb_size
        assert(self.gru_size == self.emb_size)
        
        self.hops = hops
        self.w2i = w2i
        

        self.encoder = Encoder(hops, nwords, gru_size)
        self.decoder = Decoder(emb_size, hops, gru_size, nwords)

        self.encoder_optimizer = torch.optim.Adam(self.encoder.parameters(), lr=0.001)
        self.decoder_optimizer = torch.optim.Adam(self.decoder.parameters(), lr=0.001)

        self.loss = 0
        self.ploss = 0
        self.vloss = 0
        self.n = 1
        self.batch_size = 0

    def print_loss(self):
        print_loss_avg = self.loss / self.n
        print_ploss = self.ploss / self.n
        print_vloss = self.vloss / self.n
        self.n += 1
        return 'L:{:.5f}, VL:{:.5f}, PL:{:.5f}'.format(print_loss_avg, print_vloss, print_ploss)

    def train_batch(self, input_batches, input_lengths, target_batches,
                    target_lengths, target_index, target_gate, batch_size, clip,
                    teacher_forcing_ratio, reset):
        if reset:
            self.loss = 0
            self.ploss = 0
            self.vloss = 0
            self.n = 1

        self.batch_size = input_batches.size(1)
        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()


        loss_Vocab, ploss = 0, 0

        # Run words through encoder
        decoder_hidden = self.encoder(input_batches.transpose(0,1)).unsqueeze(0)

        self.decoder.load_memory(input_batches.transpose(0, 1))

        # Prepare input and output variables
        decoder_input = Variable(torch.LongTensor([2] * self.batch_size))

        max_target_length = max(target_lengths)
        all_decoder_outputs_vocab = Variable(torch.zeros(max_target_length, self.batch_size, self.nwords))
        all_decoder_outputs_ptr = Variable(torch.zeros(max_target_length, self.batch_size, input_batches.size(0)))

        # Move new Variables to CUDA
        if USE_CUDA:
            all_decoder_outputs_vocab = all_decoder_outputs_vocab.cuda()
            all_decoder_outputs_ptr = all_decoder_outputs_ptr.cuda()
            decoder_input = decoder_input.cuda()


        for t in range(max_target_length):
            decoder_ptr, decoder_vacab, decoder_hidden = self.decoder(input_batches, decoder_input, decoder_hidden)

            all_decoder_outputs_vocab[t] = decoder_vacab
            all_decoder_outputs_ptr[t] = decoder_ptr

            decoder_input = target_batches[t]  # Chosen word is next input
            if USE_CUDA: decoder_input = decoder_input.cuda()

        # Loss calculation and backpropagation
        loss_Vocab = masked_cross_entropy(
            all_decoder_outputs_vocab.transpose(0, 1).contiguous(),  # -> batch x seq
            target_batches.transpose(0, 1).contiguous(),  # -> batch x seq
            target_lengths
        )
        ploss = masked_cross_entropy(
            all_decoder_outputs_ptr.transpose(0, 1).contiguous(),  # -> batch x seq
            target_index.transpose(0, 1).contiguous(),  # -> batch x seq
            target_lengths
        )

        loss = loss_Vocab + ploss
        loss.backward()

        # Clip gradient norms
        ec = torch.nn.utils.clip_grad_norm(self.encoder.parameters(), clip)
        dc = torch.nn.utils.clip_grad_norm(self.decoder.parameters(), clip)
        # Update parameters with optimizers
        self.encoder_optimizer.step()
        self.decoder_optimizer.step()
        self.loss += loss.item()
        self.ploss += ploss.item()
        self.vloss += loss_Vocab.item()


class Encoder(nn.Module):
    def __init__(self, hops, nwords, emb_size):
        super(Encoder, self).__init__()

        def init_weights(m):
            if type(m) == torch.nn.Embedding:
                m.weight.data=torch.normal(0.0,torch.ones(self.nwords,self.emb_size)*0.1)
                # m.weight.data.fill_(1.0)

        self.hops = hops
        self.nwords = nwords
        self.emb_size = emb_size
        self.dropout = 0.2

        #(TODO) : Initialize with word2vec
        # self.A = torch.nn.ModuleList([torch.nn.Embedding(self.nwords, self.emb_size, padding_idx=0) for h in range(self.hops)])
        # self.A.apply(init_weights)
        self.C = torch.nn.ModuleList([torch.nn.Embedding(self.nwords, self.emb_size, padding_idx=0) for h in range(self.hops+1)])
        self.C.apply(init_weights)
        # for i in range(self.hops-1):
        #     self.C[i].weight = self.A[i+1].weight
        self.soft = torch.nn.Softmax(dim=1)

    def forward(self, context):
        # (TODO): Use pack_padded_sequence
        # context : batchsize x length x 3
        # pdb.set_trace()
        size = context.size() # b x l x 3


        q = torch.zeros(size[0], self.emb_size).type(TYPEF) # initialize u # batchsize x length x emb_size
        q_list = [q]

        context = context.view(size[0],-1) # b x l*3
        for h in range(self.hops):
            m = self.C[h](context) # b x l*3 x e
            m = m.view(size[0],size[1],size[2],self.emb_size) # b x l x 3 x e
            m = torch.sum(m,2) # b x l x e
            p = torch.sum(m*q.unsqueeze(1), 2) # b x l (TODO): expand_as(m)
            attn = self.soft(p)

            c = self.C[h+1](context) # b x l*3 x e
            c = c.view(size[0],size[1],size[2],self.emb_size) # b x l x 3 x e
            c = torch.sum(c,2).squeeze(2) # b x l x e
            o = torch.bmm(attn.unsqueeze(1), c).squeeze(1)
            q = q + o
        return q










class Decoder(nn.Module):
    # def __init__(self, vocab, embedding_dim, hop, dropout, unk_mask):
    def __init__(self, emb_size, hops, gru_size, nwords):

        super(Decoder, self).__init__()
        self.nwords = nwords
        self.hops = hops
        self.emb_size = emb_size
        self.gru_size = gru_size
        def init_weights(m):
            if type(m) == torch.nn.Embedding:
                m.weight.data=torch.normal(0.0,torch.ones(self.nwords,self.emb_size)*0.1)
        self.C = torch.nn.ModuleList([torch.nn.Embedding(self.nwords, self.emb_size)\
                                      for h in range(self.hops+1)])
        self.C.apply(init_weights)
        self.soft = nn.Softmax(dim=1)
        self.lin_vocab = nn.Linear(2*emb_size,self.nwords)
        self.gru = nn.GRU(emb_size, emb_size)

    def load_memory(self, context):
        size = context.size() # b * m * 3
        self.memories = []
        context = context.view(size[0],-1)
        for hop in range(self.hops):
            m = self.C[hop](context)
            m = m.view(size[0],size[1],size[2],self.emb_size)
            m = torch.sum(m, 2)
            self.memories.append(m)
            c = self.C[hop+1](context)
            c = c.view(size[0], size[1], size[2], self.emb_size)
            c = torch.sum(c, 2)
        self.memories.append(c)

    def forward(self, context, y_, h_):
        m = self.C[0](y_).unsqueeze(0) # b * e
        _, h = self.gru(m, h_)

        q = [h.squeeze(0)]
        for hop in range(self.hops):
            p = torch.sum(self.memories[hop]*q[-1].unsqueeze(1).expand_as(self.memories[hop]), 2)
            attn = self.soft(p)
            o = torch.bmm(attn.unsqueeze(1),self.memories[hop+1]).squeeze(1)
            q.append(q[-1] + o)
            if hop==0:
                p_vocab = self.lin_vocab(torch.cat((q[0], o),1))

        p_ptr = p
        return p_ptr, p_vocab, h


