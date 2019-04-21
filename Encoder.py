import torch.nn as nn
import torch
import numpy as np
from torch.autograd import Variable

TYPE = torch.LongTensor
TYPEF = torch.FloatTensor
use_cuda = torch.cuda.is_available()
if use_cuda:
    TYPE = torch.cuda.LongTensor
    TYPEF = torch.cuda.FloatTensor

class Encoder(nn.Module):
    def __init__(self, hops, nwords, emb_size):
        super(Encoder, self).__init__()

        def init_weights(m):
            if type(m) == torch.nn.Embedding:
                m.weight.data = torch.normal(0.0, torch.ones(self.nwords, self.emb_size) * 0.1)
                # m.weight.data.fill_(1.0)

        self.hops = hops
        self.nwords = nwords
        self.emb_size = emb_size
        self.dropout = 0.2

        # (TODO) : Initialize with word2vec
        self.A = torch.nn.ModuleList(
            [torch.nn.Embedding(self.nwords, self.emb_size, padding_idx=0) for h in range(self.hops)])
        self.A.apply(init_weights)
        self.C = torch.nn.ModuleList(
            [torch.nn.Embedding(self.nwords, self.emb_size, padding_idx=0) for h in range(self.hops)])
        self.C.apply(init_weights)
        for i in range(self.hops - 1):
            self.C[i].weight = self.A[i + 1].weight
        self.soft = torch.nn.Softmax(dim=1)

    def forward(self, context):
        # (TODO): Use pack_padded_sequence
        # context : batchsize x length x 3
        # pdb.set_trace()
        size = context.size()  # b x l x 3

        if (self.training):  ### Dropout
            ones = np.ones((size[0], size[1], size[2]))
            rand_mask = np.random.binomial([np.ones((size[0], size[1]))], 1 - self.dropout)[0]
            ones[:, :, 0] = ones[:, :, 0] * rand_mask
            a = Variable(torch.Tensor(ones))
            if use_cuda: a = a.cuda()
            context = context * a.long()

        q = torch.zeros(size[0], self.emb_size).type(TYPEF)  # initialize u # batchsize x length x emb_size
        q_list = [q]

        context = context.contiguous().view(size[0], -1)  # b x l*3
        for h in range(self.hops):
            m = self.A[h](context)  # b x l*3 x e
            m = m.view(size[0], size[1], size[2], self.emb_size)  # b x l x 3 x e
            m = torch.sum(m, 2)  # b x l x e
            p = torch.sum(m * q.unsqueeze(1), 2)  # b x l (TODO): expand_as(m)
            attn = self.soft(p)

            c = self.C[h](context)  # b x l*3 x e
            c = c.view(size[0], size[1], size[2], self.emb_size)  # b x l x 3 x e
            c = torch.sum(c, 2).squeeze(2)  # b x l x e
            o = torch.bmm(attn.unsqueeze(1), c).squeeze(1)
            q = q + o
        return q
