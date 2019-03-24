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
import random
import numpy as np

USE_CUDA = torch.cuda.is_available()

TYPE = torch.LongTensor
TYPEF = torch.FloatTensor
use_cuda = torch.cuda.is_available()

if use_cuda:
    TYPE = torch.cuda.LongTensor
    TYPEF = torch.cuda.FloatTensor
    # model.cuda()

class Mem2Seq(nn.Module):
    def __init__(self, hidden_size, nwords, lr, n_layers, dropout, unk_mask):
        super(Mem2Seq, self).__init__()
        self.name = "Mem2Seq"
        self.input_size = nwords
        self.output_size = nwords
        self.hidden_size = hidden_size
        self.lr = lr
        self.n_layers = n_layers
        self.dropout = dropout
        self.unk_mask = unk_mask

        self.encoder_og = Encoder(n_layers, nwords, hidden_size)
        self.decoder = DecoderMemNN(nwords, hidden_size, n_layers, self.dropout, self.unk_mask)


        # Initialize optimizers and criterion
        self.encoder_og_optimizer = torch.optim.Adam(self.encoder_og.parameters(), lr=lr)
        self.decoder_optimizer = torch.optim.Adam(self.decoder.parameters(), lr=lr)

        self.scheduler = lr_scheduler.ReduceLROnPlateau(self.decoder_optimizer, mode='max', factor=0.5, patience=1,
                                                        min_lr=0.0001, verbose=True)
        self.criterion = nn.MSELoss()
        self.loss = 0
        self.loss_ptr = 0
        self.loss_vac = 0
        self.print_every = 1
        self.batch_size = 0
        # Move models to GPU
        if USE_CUDA:
            self.encoder.cuda()
            self.decoder.cuda()


    def print_loss(self):
        print_loss_avg = self.loss / self.print_every
        print_loss_ptr = self.loss_ptr / self.print_every
        print_loss_vac = self.loss_vac / self.print_every
        self.print_every += 1
        return 'L:{:.5f}, VL:{:.5f}, PL:{:.5f}'.format(print_loss_avg, print_loss_vac, print_loss_ptr)

    def train_batch(self, input_batches, input_lengths, target_batches,
                    target_lengths, target_index, target_gate, batch_size, clip,
                    teacher_forcing_ratio, reset):
        if reset:
            self.loss = 0
            self.loss_ptr = 0
            self.loss_vac = 0
            self.print_every = 1

        self.batch_size = input_batches.size(1)
        # Zero gradients of both optimizers
        # self.encoder_optimizer.zero_grad()
        self.encoder_og_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()
        # self.decoder_og_optimizer.zero_grad()


        loss_Vocab, loss_Ptr = 0, 0

        # Run words through encoder
        decoder_hidden = self.encoder_og(input_batches.transpose(0,1)).unsqueeze(0)

        self.decoder.load_memory(input_batches.transpose(0, 1))

        # Prepare input and output variables
        decoder_input = Variable(torch.LongTensor([2] * self.batch_size))

        max_target_length = max(target_lengths)
        all_decoder_outputs_vocab = Variable(torch.zeros(max_target_length, self.batch_size, self.output_size))
        all_decoder_outputs_ptr = Variable(torch.zeros(max_target_length, self.batch_size, input_batches.size(0)))

        # Move new Variables to CUDA
        if USE_CUDA:
            all_decoder_outputs_vocab = all_decoder_outputs_vocab.cuda()
            all_decoder_outputs_ptr = all_decoder_outputs_ptr.cuda()
            decoder_input = decoder_input.cuda()

        # Choose whether to use teacher forcing
        use_teacher_forcing = random.random() < teacher_forcing_ratio
        use_teacher_forcing = 1

        if use_teacher_forcing:
            # Run through decoder one time step at a time
            for t in range(max_target_length):
                decoder_ptr, decoder_vacab, decoder_hidden = self.decoder.ptrMemDecoder(decoder_input, decoder_hidden)

                all_decoder_outputs_vocab[t] = decoder_vacab
                all_decoder_outputs_ptr[t] = decoder_ptr

                decoder_input = target_batches[t]  # Chosen word is next input
                if USE_CUDA: decoder_input = decoder_input.cuda()
        else:
            for t in range(max_target_length):
                decoder_ptr, decoder_vacab, decoder_hidden = self.decoder.ptrMemDecoder(decoder_input, decoder_hidden)
                _, toppi = decoder_ptr.data.topk(1)
                _, topvi = decoder_vacab.data.topk(1)
                all_decoder_outputs_vocab[t] = decoder_vacab
                all_decoder_outputs_ptr[t] = decoder_ptr
                ## get the correspective word in input
                top_ptr_i = torch.gather(input_batches[:, :, 0], 0, Variable(toppi.view(1, -1))).transpose(0, 1)
                next_in = [top_ptr_i[i].item() if (toppi[i].item() < input_lengths[i] - 1) else topvi[i].item() for i in
                           range(self.batch_size)]

                decoder_input = Variable(torch.LongTensor(next_in))  # Chosen word is next input
                if USE_CUDA: decoder_input = decoder_input.cuda()

        # Loss calculation and backpropagation
        loss_Vocab = masked_cross_entropy(
            all_decoder_outputs_vocab.transpose(0, 1).contiguous(),  # -> batch x seq
            target_batches.transpose(0, 1).contiguous(),  # -> batch x seq
            target_lengths
        )
        loss_Ptr = masked_cross_entropy(
            all_decoder_outputs_ptr.transpose(0, 1).contiguous(),  # -> batch x seq
            target_index.transpose(0, 1).contiguous(),  # -> batch x seq
            target_lengths
        )

        loss = loss_Vocab + loss_Ptr
        loss.backward()

        # Clip gradient norms
        ec = torch.nn.utils.clip_grad_norm(self.encoder_og.parameters(), clip)
        dc = torch.nn.utils.clip_grad_norm(self.decoder.parameters(), clip)
        # Update parameters with optimizers
        self.encoder_og_optimizer.step()
        self.decoder_optimizer.step()
        self.loss += loss.item()
        self.loss_ptr += loss_Ptr.item()
        self.loss_vac += loss_Vocab.item()


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


# class Decoder(nn.Module):
#     def __init__(self, emb_size, hops, gru_size, nwords, batch_size):
#         super(Decoder, self).__init__()
#         self.emb_size = emb_size
#         self.gru_size = gru_size
#         self.hops = hops
#         self.nwords = nwords
#         self.gru = torch.nn.GRU(input_size=self.emb_size,
#                                 hidden_size=self.gru_size,
#                                 num_layers=1)
#         def init_weights(m):
#             if type(m) == torch.nn.Embedding:
#                 m.weight.data=torch.normal(0.0,torch.ones(self.nwords,self.emb_size)*0.1)
#                 # m.weight.data.fill_(1.0)
#         # self.A = torch.nn.ModuleList([torch.nn.Embedding(self.nwords, self.emb_size)\
#         #                               for h in range(self.hops)])
#         self.C = torch.nn.ModuleList([torch.nn.Embedding(self.nwords, self.emb_size)\
#                                       for h in range(self.hops+1)])
#         self.C.apply(init_weights)
#
#         # self.gru.apply(init_weights)
#         # for i in range(self.hops-1):
#         #     self.C[i].weight = self.A[i+1].weight
#
#         # from torch.nn import init
#         # for p in self.gru.parameters():
#         #     init.constant(p,1.0)
#
#         self.soft = torch.nn.Softmax(dim = 1)
#         self.lin_vocab = torch.nn.Linear(2*self.emb_size, self.nwords)
#         self.memories = []
#         self.dropout = 0.2
#
#
#     def load_memory(self, context):
#         size = context.size()
#         # context = context.permute(1, 0, 2)
#         self.memories = []
#         context = context.view(size[0], -1)  # b x l*3
#         for hop in range(self.hops):
#             m = self.C[hop](context) # b x l*3 x e
#             m = m.view(size[0], size[1], size[2], self.emb_size) # b x l x 3 x e
#             m = torch.sum(m, 2)  # b x l x e
#             self.memories.append(m)
#             m_ = self.C[hop+1](context) # b x l*3 x e
#             m_ = m_.view(size[0], size[1], size[2], self.emb_size) # b x l x 3 x e
#             m_ = torch.sum(m_, 2)  # b x l x e
#         self.memories.append(m_)
#         return self.memories
#
#
#
#     def forward(self, context, h_, y_): # (TODO) : Think about pack padded sequence
#         y_ = self.C[0](y_).unsqueeze(0) # 1 x b x e
#
#         _, h = self.gru(y_, h_) # 1 x b x e
#         size = context.size()
#         context = context.view(size[0], -1) # b x l*3
#
#         q = [h.squeeze(0)]
#
#         for hop in range(self.hops):
#             m = self.memories[hop]
#             q_ = q[-1].unsqueeze(1).expand_as(m)
#             p = torch.sum(m * q_, 2)  # b x l
#             attn = self.soft(p)  # b x l
#
#             c = self.memories[hop+1]
#             o = torch.bmm(attn.unsqueeze(1), c).squeeze(1) # b x e
#             q.append(q[-1] + o)
#             if hop == 0:
#                 p_vocab = self.lin_vocab(torch.cat((q[0], o),1))
#
#         p_ptr = p
#         return h, p_vocab, p_ptr







class DecoderMemNN(nn.Module):
    def __init__(self, vocab, embedding_dim, hop, dropout, unk_mask):
        super(DecoderMemNN, self).__init__()
        self.num_vocab = vocab
        self.max_hops = hop
        self.embedding_dim = embedding_dim
        self.dropout = dropout
        self.unk_mask = unk_mask
        for hop in range(self.max_hops+1):
            C = nn.Embedding(self.num_vocab, embedding_dim, padding_idx=0)
            C.weight.data.normal_(0, 0.1)
            self.add_module("C_{}".format(hop), C)
        self.C = AttrProxy(self, "C_")
        self.softmax = nn.Softmax(dim=1)
        self.W = nn.Linear(embedding_dim,1)
        self.W1 = nn.Linear(2*embedding_dim,self.num_vocab)
        self.gru = nn.GRU(embedding_dim, embedding_dim, dropout=dropout)

    def load_memory(self, story):
        story_size = story.size() # b * m * 3
        if self.unk_mask:
            if(self.training):
                ones = np.ones((story_size[0],story_size[1],story_size[2]))
                rand_mask = np.random.binomial([np.ones((story_size[0],story_size[1]))],1-self.dropout)[0]
                ones[:,:,0] = ones[:,:,0] * rand_mask
                a = Variable(torch.Tensor(ones))
                if USE_CUDA:
                    a = a.cuda()
                story = story*a.long()
        self.m_story = []
        for hop in range(self.max_hops):
            embed_A = self.C[hop](story.contiguous().view(story.size(0), -1))#.long()) # b * (m * s) * e
            embed_A = embed_A.view(story_size+(embed_A.size(-1),)) # b * m * s * e
            embed_A = torch.sum(embed_A, 2).squeeze(2) # b * m * e
            m_A = embed_A
            embed_C = self.C[hop+1](story.contiguous().view(story.size(0), -1).long())
            embed_C = embed_C.view(story_size+(embed_C.size(-1),))
            embed_C = torch.sum(embed_C, 2).squeeze(2)
            m_C = embed_C
            self.m_story.append(m_A)
        self.m_story.append(m_C)

    def ptrMemDecoder(self, enc_query, last_hidden):
        embed_q = self.C[0](enc_query) # b * e
        output, hidden = self.gru(embed_q.unsqueeze(0), last_hidden)
        temp = []
        u = [hidden[0].squeeze()]
        for hop in range(self.max_hops):
            m_A = self.m_story[hop]
            if(len(list(u[-1].size()))==1): u[-1] = u[-1].unsqueeze(0) ## used for bsz = 1.
            u_temp = u[-1].unsqueeze(1).expand_as(m_A)
            prob_lg = torch.sum(m_A*u_temp, 2)
            prob_   = self.softmax(prob_lg)
            m_C = self.m_story[hop+1]
            temp.append(prob_)
            prob = prob_.unsqueeze(2).expand_as(m_C)
            o_k  = torch.sum(m_C*prob, 1)
            if (hop==0):
                p_vocab = self.W1(torch.cat((u[0], o_k),1))
            u_k = u[-1] + o_k
            u.append(u_k)
        p_ptr = prob_lg
        return p_ptr, p_vocab, hidden



class AttrProxy(object):
    """
    Translates index lookups into attribute lookups.
    To implement some trick which able to use list of nn.Module in a nn.Module
    see https://discuss.pytorch.org/t/list-of-nn-module-in-a-nn-module/219/2
    """
    def __init__(self, module, prefix):
        self.module = module
        self.prefix = prefix

    def __getitem__(self, i):
        return getattr(self.module, self.prefix + str(i))