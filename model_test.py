import torch
import numpy as np

from torch import nn
from masked_cross_entropy import *

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
from data import find_entities

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

        self.optim_enc = torch.optim.Adam(self.encoder.parameters(), lr=0.001)
        self.optim_dec = torch.optim.Adam(self.decoder.parameters(), lr=0.001)
        self.scheduler = lr_scheduler.ReduceLROnPlateau(self.optim_dec, mode='max', factor=0.5, patience=1,
                                                        min_lr=0.0001, verbose=True)

        self.cross_entropy = torch.nn.CrossEntropyLoss()  # masked_cross_entropy

        if use_cuda:
            self.cross_entropy = self.cross_entropy.cuda()
            self.encoder = self.encoder.cuda()
            self.decoder = self.decoder.cuda()
        self.loss = 0
        self.ploss = 0
        self.vloss = 0
        self.n = 1
        self.batch_size = 0
        self.dropout = 0.2

        self.plot_data = {
            'train': {
                'loss': [],
                'vocab_loss': [],
                'ptr_loss': [],
            },
            'val':{
                'loss': [],
                'vocab_loss': [],
                'ptr_loss': [],
                'wer': [],
                'acc': []
            },
            'test': {
                'loss': [],
                'vocab_loss': [],
                'ptr_loss': [],
            },
        }

    def print_loss(self):
        print_loss_avg = self.loss / self.n
        print_ploss = self.ploss / self.n
        print_vloss = self.vloss / self.n
        self.n += 1
        return 'L:{:.5f}, VL:{:.5f}, PL:{:.5f}'.format(print_loss_avg, print_vloss, print_ploss)

    def losses(self):
        return self.loss / self.n, self.ploss / self.n, self.vloss / self.n

    def train_batch(self, context, responses, index, sentinel, new_epoch, context_lengths, target_lengths, clip_grads):

        # (TODO): remove transpose
        if new_epoch:  # (TODO): Change this part
            self.loss = 0
            self.ploss = 0
            self.vloss = 0
            self.n = 1

        context = context.type(TYPE)
        responses = responses.type(TYPE)
        index = index.type(TYPE)
        sentinel = sentinel.type(TYPE)

        self.optim_enc.zero_grad()
        self.optim_dec.zero_grad()

        h = self.encoder(context.transpose(0, 1))
        self.decoder.load_memory(context.transpose(0, 1))
        y = torch.from_numpy(np.array([2] * context.size(1), dtype=int)).type(TYPE)
        y_len = 0

        h = h.unsqueeze(0)
        output_vocab = torch.zeros(max(target_lengths), context.size(1), self.nwords)
        output_ptr = torch.zeros(max(target_lengths), context.size(1), context.size(0))
        while y_len < responses.size(0):  # TODO: Add EOS condition
            p_ptr, p_vocab, h = self.decoder(context, y, h)
            output_vocab[y_len] = p_vocab
            output_ptr[y_len] = p_ptr

            y = responses[y_len].type(TYPE)
            y_len += 1

        # print(loss)
        mask_v = torch.ones(output_vocab.size())
        mask_p = torch.ones(output_ptr.size())

        for i in range(responses.size(1)):
            mask_v[target_lengths[i]:, i, :] = 0
            mask_p[target_lengths[i]:, i, :] = 0


        loss_v = self.cross_entropy(output_vocab.contiguous().view(-1, self.nwords), responses.cpu().contiguous().view(-1))
        loss_ptr = self.cross_entropy(output_ptr.contiguous().view(-1, context.size(0)), index.cpu().contiguous().view(-1))

        loss = loss_ptr + loss_v

        loss.backward()
        ec = torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), 10.0)
        dc = torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), 10.0)
        self.optim_enc.step()
        self.optim_dec.step()

        self.loss += loss.item()
        self.vloss += loss_v.item()
        self.ploss += loss_ptr.item()

    def save_models(self, path):
        import os
        torch.save(self.encoder.state_dict(), os.path.join(path, 'encoder.pth'))
        torch.save(self.decoder.state_dict(), os.path.join(path, 'decoder.pth'))

    def load_models(self, path: object = '.') -> object:
        import os
        self.encoder.load_state_dict(os.path.join(path, 'encoder.pth'))
        self.decoder.load_state_dict(os.path.join(path, 'decoder.pth'))

    # def evaluate(self, context, response):
    #     #assert (context.size(0) == 1)
    #     ctx = context.transpose(0,1)
    #     res_trans = response.transpose(0,1).type(TYPE)
    #     self.decoder.load_memory(ctx)
    #     ctx = ctx.type(TYPE)
    #     context = context.type(TYPE)
    #     response = response.type(TYPE)
    #
    #     h = self.encoder(ctx)
    #     y = torch.from_numpy(np.array([2] * ctx.size(0), dtype=int)).type(TYPE)
    #     y_len = 0
    #
    #     loss = 0
    #     loss_v = 0
    #     loss_ptr = 0
    #     h = h.unsqueeze(0)
    #     output = np.full((ctx.size(0), res_trans.size(1)),-1 )
    #     mask = np.ones(ctx.size(0),dtype=np.int32)
    #     correct_words = 0
    #     total_words = 0
    #     while y_len < res_trans.size(1):  #
    #         p_ptr, p_vocab, h = self.decoder(ctx, y, h)
    #         p_ptr_argmax = torch.argmax(p_ptr, dim=1).data.numpy()
    #         p_vocab_argmax = torch.argmax(p_vocab, dim=1).data.numpy()
    #         #output = np.zeros(p_ptr_argmax.shape[0], dtype=np.int32)
    #         for i, max_idx in enumerate(p_ptr_argmax):
    #             if max_idx < ctx.size(1): #Not a sentinel
    #                 output[i, y_len] = ctx[i][max_idx][0].item()
    #             else:
    #                 output[i, y_len] = p_vocab_argmax[i]
    #         correct_words += sum(np.multiply(mask,output[:,y_len]==res_trans[:, y_len].data.numpy()))
    #
    #         y = res_trans[:, y_len].type(TYPE)
    #         total_words += sum(mask)
    #         mask_candidate = output[:,y_len] == self.w2i['<eos>']
    #         for m_idx, m in enumerate(mask_candidate):
    #             if m == True:
    #                 mask[max_idx] = 0
    #         y_len += 1
    #
    #     return correct_words/total_words

    def evaluate_batch(self, batch_size, input_batches, input_lengths, target_batches, target_lengths, target_index,
                       target_gate, src_plain):


        # Set to not-training mode to disable dropout
        self.encoder.train(False)
        self.decoder.train(False)
        # Run words through encoder
        decoder_hidden = self.encoder(input_batches.transpose(0,1)).unsqueeze(0)
        self.decoder.load_memory(input_batches.transpose(0, 1))

        # Prepare input and output variables
        decoder_input = Variable(torch.LongTensor([2] * batch_size))

        decoded_words = []
        all_decoder_outputs_vocab = Variable(torch.zeros(max(target_lengths), batch_size, self.nwords))
        all_decoder_outputs_ptr = Variable(torch.zeros(max(target_lengths), batch_size, input_batches.size(0)))
        # all_decoder_outputs_gate = Variable(torch.zeros(self.max_r, batch_size))
        # Move new Variables to CUDA

        if use_cuda:
            all_decoder_outputs_vocab = all_decoder_outputs_vocab.cuda()
            all_decoder_outputs_ptr = all_decoder_outputs_ptr.cuda()
            # all_decoder_outputs_gate = all_decoder_outputs_gate.cuda()
            decoder_input = decoder_input.cuda()

        p = []
        for elm in src_plain:
            elm_temp = [word_triple[0] for word_triple in elm]
            p.append(elm_temp)

        self.from_whichs = []
        acc_gate, acc_ptr, acc_vac = 0.0, 0.0, 0.0
        # Run through decoder one time step at a time
        for t in range(max(target_lengths)):
            decoder_ptr, decoder_vacab, decoder_hidden = self.decoder(input_batches, decoder_input, decoder_hidden)
            all_decoder_outputs_vocab[t] = decoder_vacab
            topv, topvi = decoder_vacab.data.topk(1)
            all_decoder_outputs_ptr[t] = decoder_ptr
            topp, toppi = decoder_ptr.data.topk(1)
            top_ptr_i = torch.gather(input_batches[:, :, 0], 0, Variable(toppi.view(1, -1))).transpose(0, 1)
            next_in = [top_ptr_i[i].item() if (toppi[i].item() < input_lengths[i] - 1) else topvi[i].item() for i in
                       range(batch_size)]

            decoder_input = Variable(torch.LongTensor(next_in))  # Chosen word is next input
            if use_cuda: decoder_input = decoder_input.cuda()

            temp = []
            from_which = []
            for i in range(batch_size):
                if (toppi[i].item() < len(p[i]) - 1):
                    temp.append(p[i][toppi[i].item()])
                    from_which.append('p')
                else:
                    if target_index[t][i] != toppi[i].item():
                        self.incorrect_sentinel += 1
                    ind = topvi[i].item()
                    if ind == 3:
                        temp.append('<eos>')
                    else:
                        temp.append(self.i2w[ind])
                    from_which.append('v')
            decoded_words.append(temp)
            self.from_whichs.append(from_which)
        self.from_whichs = np.array(self.from_whichs)

        loss_v = self.cross_entropy(all_decoder_outputs_vocab.contiguous().view(-1, self.nwords), target_batches.contiguous().view(-1))
        loss_ptr = self.cross_entropy(all_decoder_outputs_ptr.contiguous().view(-1, input_batches.size(0)), target_index.contiguous().view(-1))

        loss = loss_ptr + loss_v

        self.loss += loss.item()
        self.vloss += loss_v.item()
        self.ploss += loss_ptr.item()
        self.n += 1

        # Set back to training mode
        self.encoder.train(True)
        self.decoder.train(True)
        return decoded_words  # , acc_ptr, acc_vac

    def evaluate(self, dev, avg_best, kb_entries, i2w):
        self.loss = 0
        self.ploss = 0
        self.vloss = 0
        self.n = 1

        self.incorrect_sentinel = 0
        self.i2w = i2w

        def wer(r, h):
            """
            This is a function that calculate the word error rate in ASR.
            You can use it like this: wer("what is it".split(), "what is".split())
            """
            # build the matrix
            d = np.zeros((len(r) + 1) * (len(h) + 1), dtype=np.uint8).reshape((len(r) + 1, len(h) + 1))
            for i in range(len(r) + 1):
                for j in range(len(h) + 1):
                    if i == 0:
                        d[0][j] = j
                    elif j == 0:
                        d[i][0] = i
            for i in range(1, len(r) + 1):
                for j in range(1, len(h) + 1):
                    if r[i - 1] == h[j - 1]:
                        d[i][j] = d[i - 1][j - 1]
                    else:
                        substitute = d[i - 1][j - 1] + 1
                        insert = d[i][j - 1] + 1
                        delete = d[i - 1][j] + 1
                        d[i][j] = min(substitute, insert, delete)
            result = float(d[len(r)][len(h)]) / len(r) * 100
            # result = str("%.2f" % result) + "%"
            return result


        logging.info("STARTING EVALUATION")
        acc_avg = 0.0
        wer_avg = 0.0
        bleu_avg = 0.0
        acc_P = 0.0
        acc_V = 0.0
        ref = []
        hyp = []
        ref_s = ""
        hyp_s = ""
        dialog_acc_dict = {}

        global_entity_list = kb_entries

        pbar = tqdm(enumerate(dev), total=len(dev))
        for j, data_dev in pbar:

            words = self.evaluate_batch(len(data_dev[1]), data_dev[0].transpose(0,1), data_dev[4], data_dev[1].transpose(0,1), data_dev[5],
                                        data_dev[2].transpose(0,1), data_dev[3].transpose(0,1), data_dev[7])

            acc = 0
            w = 0
            temp_gen = []

            for i, row in enumerate(np.transpose(words)):
                st = ''
                for e in row:
                    if e == '<eos>':
                        break
                    else:
                        st += e + ' '
                temp_gen.append(st)
                correct = data_dev[8][i]
                ### compute F1 SCORE
                st = st.lstrip().rstrip()
                correct = correct.lstrip().rstrip()


                if data_dev[6][i] not in dialog_acc_dict.keys():
                    dialog_acc_dict[data_dev[6][i].item()] = []
                if (correct == st):
                    acc += 1
                    dialog_acc_dict[data_dev[6][i].item()].append(1)
                else:
                    dialog_acc_dict[data_dev[6][i].item()].append(0)


                w += wer(correct, st)
                ref.append(str(correct))
                hyp.append(str(st))
                ref_s += str(correct) + "\n"
                hyp_s += str(st) + "\n"

            acc_avg += acc / float(len(data_dev[1]))
            wer_avg += w / float(len(data_dev[1]))
            pbar.set_description("R:{:.4f},W:{:.4f},I:{:.4f}".format(acc_avg / float(len(dev)),
                                                            wer_avg / float(len(dev)), self.incorrect_sentinel / float(len(dev))))

            self.plot_data['val']['acc'].append(acc_avg / float(len(dev)))
            self.plot_data['val']['wer'].append(wer_avg / float(len(dev)))

            self.plot_data['val']['loss'].append(self.losses()[0])
            self.plot_data['val']['vocab_loss'].append(self.losses()[1])
            self.plot_data['val']['ptr_loss'].append(self.losses()[2])

        # dialog accuracy

        dia_acc = 0
        for k in dialog_acc_dict.keys():
            if len(dialog_acc_dict[k]) == sum(dialog_acc_dict[k]):
                dia_acc += 1
        logging.info("Dialog Accuracy:\t" + str(dia_acc * 1.0 / len(dialog_acc_dict.keys())))


        acc_avg = acc_avg / float(len(dev))

        if (acc_avg >= avg_best):
            return acc_avg

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
