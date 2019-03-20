import torch
import numpy as np

from torch import nn


class Model(nn.Module):
    def __init__(self, hops, nwords, emb_size, gru_size, batch_size, w2i):
        super(Model, self).__init__()

        self.hops = hops
        self.nwords = nwords
        self.emb_size = emb_size
        self.gru_size = gru_size
        self.batch_size = batch_size
        self.w2i = w2i

        self.encoder = Encoder(self.hops, self.nwords, self.emb_size)
        self.decoder = Decoder(self.emb_size, self.hops, self.gru_size, self.nwords, self.batch_size)

        self.optim = torch.optim.Adam(list(self.encoder.parameters()) + list(self.decoder.parameters()))
        self.cross_entropy = torch.nn.CrossEntropyLoss()

        self.loss = 0
        self.loss_vocab = 0
        self.loss_ptr = 0
        self.acc = 0

    def train(self, context, responses, index, sentinel):

        h = self.encoder(context)
        y = torch.from_numpy(np.array([3]*self.batch_size, dtype=int)).type(torch.LongTensor)
        y_len = 0

        loss = 0
        loss_v = 0
        loss_ptr = 0
        while y_len < responses.size(1): # TODO: Add EOS condition
            h, p_vocab, p_ptr = self.decoder(context, h, y)
            loss_v = self.cross_entropy(p_vocab, responses[:, y_len])
            loss_ptr = self.cross_entropy(p_ptr, index[:, y_len])
            loss += loss_v + loss_ptr

            y_len += 1
            y = responses[:, y_len - 1].type(torch.LongTensor)

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()



class Encoder(nn.Module):
    def __init__(self, hops, nwords, emb_size):
        super(Encoder, self).__init__()
        def init_weights(m):
            if type(m) == torch.nn.Embedding:
                m.weight.data.fill_(1.0) # intiialize properly!!!

        self.hops = hops
        self.nwords = nwords
        self.emb_size = emb_size

        #(TODO) : Initialize with word2vec 
        self.A = torch.nn.ModuleList([torch.nn.Embedding(self.nwords, self.emb_size) for h in range(self.hops)])
        self.A.apply(init_weights)
        self.C = torch.nn.ModuleList([torch.nn.Embedding(self.nwords, self.emb_size) for h in range(self.hops)])
        self.C.apply(init_weights)
        for i in range(self.hops-1):
            self.C[i].weight = self.A[i+1].weight

    def forward(self, context):
        # (TODO): Use pack_padded_sequence
        # context : batchsize x length x 3
        # pdb.set_trace()
        size = context.size()
        q = torch.zeros(size[0], self.emb_size) # initialize u # batchsize x length x emb_size
        q_list = [q] 

        context = context.view(size[0],-1) # batchsize x length*3
        # context = torch.sum(context,2) # batchsize x length
        for h in range(self.hops):
            A = self.A[h](context) # batchsize x length*3 x emb_size
            A = A.view(size[0],size[1],size[2],self.emb_size) # batchsize x length x 3 x emb_size
            A = torch.sum(A,2) # batchsize x length x emb_size
            p = torch.sum(A*q.unsqueeze(1), 2) # batchsize x length
            attn = torch.nn.functional.softmax(p, 1) # batchsize x length

            C = self.C[h](context) # batchsize x length*3 x emb_size
            C = C.view(size[0],size[1],size[2],self.emb_size) # batchsize x length x 3 x emb_size
            C = torch.sum(C,2) # batchsize x length x emb_size
            attn = attn.unsqueeze(2).expand(size[0],size[1],self.emb_size)
            o = C*attn # batchsize x length x emb_size
            o = torch.sum(o,1) # batchsize x emb_size
            q += o
        return o
        


class Decoder(nn.Module):
    def __init__(self, emb_size, hops, gru_size, nwords, batch_size):
        super(Decoder, self).__init__()
        self.emb_size = emb_size
        self.gru_size = gru_size
        self.hops = hops
        self.nwords = nwords
        self.gru = torch.nn.GRU(input_size=self.emb_size,
                                hidden_size=self.gru_size,
                                num_layers=1)
        self.A = torch.nn.ModuleList([torch.nn.Embedding(self.nwords, self.emb_size)\
                                      for h in range(self.hops)])
        self.C = torch.nn.ModuleList([torch.nn.Embedding(self.nwords, self.emb_size)\
                                      for h in range(self.hops)])

        for i in range(self.hops-1):
            self.C[i].weight = self.A[i+1].weight
        self.soft = torch.nn.Softmax(dim = 2)
        self.lin_vocab = torch.nn.Linear(2*self.emb_size, self.nwords)

    def forward(self, context, h_, y_):
        y_ = self.A[0](y_).unsqueeze(0)
        _, h = self.gru(y_, h_.unsqueeze(0)) #(seq_len, batch, input_size)
        size = context.size()
        context = context.view(size[0], -1)

        q = torch.Tensor()
        q.data = h.clone()
        o1 = torch.Tensor()
        for hop in range(self.hops):
            A = self.A[hop](context)
            A = A.view(size[0], size[1], size[2], self.emb_size)
            A = torch.sum(A, 2)  # batchsize x length x emb_size
            p = torch.sum(A * q.unsqueeze(1), 2)  # batchsize x length
            attn = torch.nn.functional.softmax(p, 1)  # batchsize x length

            C = self.C[hop](context)  # batchsize x length*3 x emb_size
            C = C.view(size[0], size[1], size[2], self.emb_size)  # batchsize x length x 3 x emb_size
            C = torch.sum(C, 2)  # batchsize x length x emb_size
            attn = attn.unsqueeze(2).expand(size[0], size[1], self.emb_size)
            o = C * attn  # batchsize x length x emb_size
            q += o
            if hop == 0:
                o1.data = o.clone()

        p_vocab = self.soft(self.lin_vocab(torch.cat((h, o1),1)))

        p_ptr = attn

        return h, p_vocab, p_ptr
