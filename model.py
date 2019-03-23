import torch
import numpy as np

from torch import nn
from masked_cross_entropy import*

TYPE = torch.LongTensor
TYPEF = torch.FloatTensor
use_cuda = torch.cuda.is_available()

if use_cuda:
    TYPE = torch.cuda.LongTensor
    TYPEF = torch.cuda.FloatTensor
    # model.cuda()


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

        self.optim_enc = torch.optim.Adam(self.encoder.parameters(), lr = 0.001)
        self.optim_dec = torch.optim.Adam(self.decoder.parameters(), lr=0.001)

        self.cross_entropy = masked_cross_entropy

        self.loss = 0
        self.loss_vocab = 0
        self.loss_ptr = 0
        self.acc = 0

        self.n = 1

    def train(self, context, responses, index, sentinel, new_epoch, context_lengths, target_lengths, clip):

        if new_epoch: # (TODO): Change this part
            self.loss = 0
            self.loss_ptr = 0
            self.loss_vocab = 0
            self.n = 1

        # with torch.autograd.set_detect_anomaly(True):
        # print(i)
        context = context.type(TYPE)
        responses = responses.type(TYPE)
        index = index.type(TYPE)
        sentinel = sentinel.type(TYPE)

        self.optim_enc.zero_grad()
        self.optim_dec.zero_grad()

        # Encoder
        h = self.encoder(context)
        # h_fake = self.encoder.forward_fake(context)
        # print(torch.all(torch.eq(h, h_fake)))

        # Load memory
        self.decoder.load_memory(context)

        # self.decoder.load_memory_fake(context)



        y = torch.from_numpy(np.array([2]*context.size(0), dtype=int)).type(TYPE)
        y_len = 0

        loss = 0
        loss_v = 0
        loss_ptr = 0
        h = h.unsqueeze(0)
        output_vocab = torch.zeros(max(target_lengths), context.size(0), len(self.w2i))
        output_ptr = torch.zeros(max(target_lengths), context.size(0), context.size(1))



        while y_len < responses.size(1): # TODO: Add EOS condition
            h, p_vocab, p_ptr = self.decoder(context, h, y)

            # h_f, p_v_f, p_p_f = self.decoder.forward_fake(y,h)
            # print(torch.all(torch.eq(h, h_f)))
            # print(torch.all(torch.eq(p_vocab, p_v_f)))
            # print(torch.all(torch.eq(p_ptr, p_p_f)))

            output_vocab[y_len] = p_vocab
            output_ptr[y_len] = p_ptr

            # loss += loss_v + loss_ptr


            y = responses[:, y_len].type(TYPE)
            y_len += 1
        # print(loss)
        loss_v = self.cross_entropy(output_vocab, responses, target_lengths)
        loss_ptr = self.cross_entropy(output_ptr, index, target_lengths)
        loss = loss_ptr + loss_v

        loss.backward()

        # Clip gradient norms
        ec = torch.nn.utils.clip_grad_norm(self.encoder.parameters(), clip)
        dc = torch.nn.utils.clip_grad_norm(self.decoder.parameters(), clip)


        self.optim_enc.step()
        self.optim_dec.step()

        self.loss += loss.item()
        self.loss_vocab += loss_v.item()
        self.loss_ptr += loss_ptr.item()

    def evaluate(self, context, response):
        assert (context.size(0) == 1)

        context = context.type(TYPE)
        response = response.type(TYPE)

        h = self.encoder(context)
        y = torch.from_numpy(np.array([2] * context.size(0), dtype=int)).type(TYPE)
        y_len = 0

        loss = 0
        loss_v = 0
        loss_ptr = 0
        h = h.unsqueeze(0)
        output = []
        correct_words = 0
        while y_len < response.size(1):  #
            h, p_vocab, p_ptr = self.decoder(context, h, y)
            if p_ptr.item() < context.size(1):
                output.append(context[0][p_ptr][0].item())
            else:
                output.append(p_vocab.argmax())
            correct_words += int(output[-1]==response[0][y_len])
            y_len += 1
            y = response[:, y_len - 1].type(TYPE)

            if output[-1] == self.w2i['<eos>']:
                break

        return correct_words/response.size(1)


        # loss = loss_ptr + loss_v


    def show_loss(self):
        L = self.loss / self.n
        L_P = self.loss_ptr / self.n
        L_V = self.loss_vocab / self.n
        self.n += 1
        return 'loss: '+str(L)+', vloss: '+str(L_V)+', ploss: '+str(L_P)+', n: '+str(self.n)

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
            # attn = torch.nn.functional.softmax(p, 1) # b x l
            attn = self.soft(p)

            c = self.C[h+1](context) # b x l*3 x e
            c = c.view(size[0],size[1],size[2],self.emb_size) # b x l x 3 x e
            c = torch.sum(c,2).squeeze(2) # b x l x e
            # attn2 = attn.unsqueeze(2).expand(size[0],size[1],self.emb_size) # b x l x e
            # o2 = c*attn2 # b x l x e
            # o2 = torch.sum(o2,1) # b x e
            o = torch.bmm(attn.unsqueeze(1), c).squeeze(1)
            # print(torch.all(torch.eq(o, o2)))
            # q = q + o2 # b x e
            q = q + o
        return q

        


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
        def init_weights(m):
            if type(m) == torch.nn.Embedding:
                m.weight.data=torch.normal(0.0,torch.ones(self.nwords,self.emb_size)*0.1)
                # m.weight.data.fill_(1.0)
        # self.A = torch.nn.ModuleList([torch.nn.Embedding(self.nwords, self.emb_size)\
        #                               for h in range(self.hops)])
        self.C = torch.nn.ModuleList([torch.nn.Embedding(self.nwords, self.emb_size)\
                                      for h in range(self.hops+1)])
        self.C.apply(init_weights)

        # self.gru.apply(init_weights)
        # for i in range(self.hops-1):
        #     self.C[i].weight = self.A[i+1].weight

        from torch.nn import init
        for p in self.gru.parameters():
            init.constant(p,1.0)

        self.soft = torch.nn.Softmax(dim = 1)
        self.lin_vocab = torch.nn.Linear(2*self.emb_size, self.nwords)
        self.memories = []

    def load_memory(self, context):
        size = context.size()
        # context = context.permute(1, 0, 2)
        self.memories = []
        context = context.view(size[0], -1)  # b x l*3
        for hop in range(self.hops):
            m = self.C[hop](context) # b x l*3 x e
            m = m.view(size[0], size[1], size[2], self.emb_size) # b x l x 3 x e
            m = torch.sum(m, 2)  # b x l x e
            self.memories.append(m)
            m_ = self.C[hop+1](context) # b x l*3 x e
            m_ = m_.view(size[0], size[1], size[2], self.emb_size) # b x l x 3 x e
            m_ = torch.sum(m_, 2)  # b x l x e
        self.memories.append(m_)

    def forward(self, context, h_, y_): # (TODO) : Think about pack padded sequence
        y_ = self.C[0](y_).unsqueeze(0) # 1 x b x e

        _, h = self.gru(y_, h_) # 1 x b x e
        size = context.size()
        context = context.view(size[0], -1) # b x l*3

        q = [h.squeeze(0)]

        for hop in range(self.hops):
            p = torch.sum(self.memories[hop] * q[-1].unsqueeze(1).expand_as(self.memories[hop]), 2)  # b x l
            attn = self.soft(p)  # b x l

            o = torch.bmm(attn.unsqueeze(1), self.memories[hop+1]).squeeze(1) # b x e
            q.append(q[-1] + o)
            if hop == 0:
                p_vocab = self.lin_vocab(torch.cat((q[0], o),1))

        p_ptr = attn
        return h, p_vocab, p_ptr

