import torch

from torch import nn
import pdb


class Model(nn.Module):
    def __init__(self):
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, *input):
        pass


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
        self.soft = torch.nn.Softmax(dim = 2)
        

    def forward(self, context):
        # context : batchsize x length x 3
        pdb.set_trace()
        size = context.size()

        q = torch.zeros(size[0], self.emb_size) # initialize u # batchsize x length x emb_size
        q_list = [q] 

        

        context = context.view(size[0],-1) # batchsize x length*3
        # context = torch.sum(context,2) # batchsize x length
        for h in range(self.hops):
            A = self.A[h](context) # batchsize x length*3 x emb_size
            A = A.view(size[0],size[1],size[2],self.emb_size) # batchsize x length x 3 x emb_size
            A = torch.sum(A,2) # batchsize x length x emb_size
            p = torch.sum(A*q,2) # batchsize x length
            attn = self.soft(p) # batchsize x length

            C = self.C[h](context) # batchsize x length*3 x emb_size
            C = C.view(size[0],size[1],size[2],self.emb_size) # batchsize x length x 3 x emb_size
            C = torch.sum(C,2) # batchsize x length x emb_size
            attn = attn.unsqueeze(1).expand(size[0],size[1],self.emb_size)
            o = C*attn # batchsize x length x emb_size
            o = torch.sum(o,1) # batchsize x emb_size
            q += o
            q_list.append(q)
        return q
        


class Decoder(nn.Module):
    def __init__(self):
        pass

    def forward(self):
        pass