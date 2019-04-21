import torch.nn as nn
import torch
import numpy as np
from torch.autograd import Variable
use_cuda = torch.cuda.is_available()

class Decoder(nn.Module):
    # def __init__(self, vocab, embedding_dim, hop, dropout, unk_mask):
    def __init__(self, emb_size, hops, gru_size, nwords):

        super(Decoder, self).__init__()
        self.nwords = nwords
        self.hops = hops
        self.emb_size = emb_size
        self.gru_size = gru_size
        self.dropout = 0.2

        def init_weights(m):
            if type(m) == torch.nn.Embedding:
                m.weight.data = torch.normal(0.0, torch.ones(self.nwords, self.emb_size) * 0.1)

        self.A = torch.nn.ModuleList(
            [torch.nn.Embedding(self.nwords, self.emb_size, padding_idx=0) for h in range(self.hops)])
        self.A.apply(init_weights)
        self.C = torch.nn.ModuleList(
            [torch.nn.Embedding(self.nwords, self.emb_size, padding_idx=0) for h in range(self.hops)])
        self.C.apply(init_weights)
        for i in range(self.hops - 1):
            self.C[i].weight = self.A[i + 1].weight

        self.soft = nn.Softmax(dim=1)
        self.lin_vocab = nn.Linear(2 * emb_size, self.nwords)
        self.gru = nn.GRU(emb_size, emb_size, dropout = self.dropout)

    def load_memory(self, context):
        size = context.size()  # b * m * 3

        if (self.training):
            ones = np.ones((size[0], size[1], size[2]))
            rand_mask = np.random.binomial([np.ones((size[0], size[1]))], 1 - self.dropout)[0]
            ones[:, :, 0] = ones[:, :, 0] * rand_mask
            a = Variable(torch.Tensor(ones))
            if use_cuda:
                a = a.cuda()
            context = context * a.long()

        self.memories = []
        context = context.view(size[0], -1)
        for hop in range(self.hops):
            m = self.A[hop](context)
            m = m.view(size[0], size[1], size[2], self.emb_size)
            m = torch.sum(m, 2)
            self.memories.append(m)
            c = self.C[hop](context)
            c = c.view(size[0], size[1], size[2], self.emb_size)
            c = torch.sum(c, 2)
        self.memories.append(c)

    def forward(self, context, y_, h_):
        m = self.C[0](y_).unsqueeze(0)  # b * e
        _, h = self.gru(m, h_)

        q = [h.squeeze(0)]
        for hop in range(self.hops):
            p = torch.sum(self.memories[hop] * q[-1].unsqueeze(1).expand_as(self.memories[hop]), 2)
            attn = self.soft(p)
            o = torch.bmm(attn.unsqueeze(1), self.memories[hop + 1]).squeeze(1)
            q.append(q[-1] + o)
            if hop == 0:
                p_vocab = self.lin_vocab(torch.cat((q[0], o), 1))

        p_ptr = p
        return p_ptr, p_vocab, h
