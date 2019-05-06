import torch
from torch.optim import lr_scheduler
from torch.autograd import Variable
from runner_base import ExperimentRunnerBase
from Encoder import Encoder
from Decoder import Decoder
import numpy as np
import os.path


class SplitMemRunner(ExperimentRunnerBase):
    def __init__(self, args):
        super(SplitMemRunner, self).__init__(args)
        self.gru_size = 128
        self.emb_size = 128
        self.dropout = 0.2
        self.hops = 3
        assert (self.gru_size == self.emb_size)

        self.encoder = Encoder(self.hops, self.nwords, self.gru_size)
        self.profile_encoder = Encoder(self.hops, self.nwords, self.gru_size)
        self.decoder = Decoder(self.emb_size, self.hops, self.gru_size, self.nwords)

        self.optim_enc = torch.optim.Adam(self.encoder.parameters(), lr=0.001)
        self.optim_enc_profile = torch.optim.Adam(self.profile_encoder.parameters(), lr=0.001)
        self.optim_dec = torch.optim.Adam(self.decoder.parameters(), lr=0.001)
        if self.loss_weighting:
            self.optim_loss_weights = torch.optim.Adam([self.loss_weights], lr=0.0001)
        self.scheduler = lr_scheduler.ReduceLROnPlateau(self.optim_dec, mode='max', factor=0.5, patience=1,
                                                        min_lr=0.0001, verbose=True)

        if self.use_cuda:
            self.cross_entropy = self.cross_entropy.cuda()
            self.encoder = self.encoder.cuda()
            self.decoder = self.decoder.cuda()
            self.profile_encoder = self.profile_encoder.cuda()


    def train_batch_wrapper(self, batch, new_epoch, clip_grads):
        context = batch[0].transpose(0, 1)
        responses = batch[1].transpose(0, 1)
        index = batch[2].transpose(0, 1)
        sentinel = batch[3].transpose(0, 1)
        context_lengths = batch[4]
        target_lengths = batch[5]
        profile_memory = batch[9].transpose(0, 1)
        return self.train_batch(context, responses, index, sentinel, new_epoch, context_lengths, target_lengths, clip_grads,
                    profile_memory)

    def train_batch(self, context, responses, index, sentinel, new_epoch, context_lengths, target_lengths, clip_grads,
                    profile_memory):

        # (TODO): remove transpose
        if new_epoch:  # (TODO): Change this part
            self.loss = 0
            self.ploss = 0
            self.vloss = 0
            self.n = 1

        context = context.type(self.TYPE)
        responses = responses.type(self.TYPE)
        index = index.type(self.TYPE)
        sentinel = sentinel.type(self.TYPE)

        self.optim_enc.zero_grad()
        self.optim_dec.zero_grad()
        self.optim_enc_profile.zero_grad()
        if self.loss_weighting:
            self.optim_loss_weights.zero_grad()

        h_context = self.encoder(context.transpose(0, 1))
        h_profile = self.profile_encoder(profile_memory.transpose(0, 1))

        h = h_context + h_profile

        self.decoder.load_memory(context.transpose(0, 1))
        y = torch.from_numpy(np.array([2] * context.size(1), dtype=int)).type(self.TYPE)
        y_len = 0

        h = h.unsqueeze(0)
        output_vocab = torch.zeros(max(target_lengths), context.size(1), self.nwords)
        output_ptr = torch.zeros(max(target_lengths), context.size(1), context.size(0))
        if self.use_cuda:
            output_vocab = output_vocab.cuda()
            output_ptr = output_ptr.cuda()
        while y_len < responses.size(0):  # TODO: Add EOS condition
            p_ptr, p_vocab, h = self.decoder(context, y, h)
            output_vocab[y_len] = p_vocab
            output_ptr[y_len] = p_ptr

            y = responses[y_len].type(self.TYPE)
            y_len += 1

        # print(loss)
        mask_v = torch.ones(output_vocab.size())
        mask_p = torch.ones(output_ptr.size())
        if self.use_cuda:
            mask_p = mask_p.cuda()
            mask_v = mask_v.cuda()
        for i in range(responses.size(1)):
            mask_v[target_lengths[i]:, i, :] = 0
            mask_p[target_lengths[i]:, i, :] = 0

        loss_v = self.cross_entropy(output_vocab.contiguous().view(-1, self.nwords),
                                    responses.contiguous().view(-1))

        loss_ptr = self.cross_entropy(output_ptr.contiguous().view(-1, context.size(0)),
                                      index.contiguous().view(-1))
        if self.loss_weighting:
            loss = loss_ptr / (2 * self.loss_weights[0] * self.loss_weights[0]) + loss_v / (
                        2 * self.loss_weights[1] * self.loss_weights[1]) + \
                   torch.log(self.loss_weights[0] * self.loss_weights[1])
        else:
            loss = loss_ptr + loss_v

        loss.backward()
        ec = torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), 10.0)
        dc = torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), 10.0)
        ec2 = torch.nn.utils.clip_grad_norm_(self.profile_encoder.parameters(), 10.0)
        self.optim_enc.step()
        self.optim_dec.step()
        self.optim_enc_profile.step()
        if self.loss_weighting:
            self.optim_loss_weights.step()

        self.loss += loss.item()
        self.vloss += loss_v.item()
        self.ploss += loss_ptr.item()

        return loss.item(), loss_v.item(), loss_ptr.item()

    def evaluate_batch(self, batch_size, input_batches, input_lengths, target_batches, target_lengths, target_index,
                       target_gate, src_plain, profile_memory=None):

        # Set to not-training mode to disable dropout
        self.encoder.train(False)
        self.decoder.train(False)
        self.profile_encoder.train(False)
        # Run words through encoder
        decoder_hidden = self.encoder(input_batches.transpose(0, 1)).unsqueeze(0) + self.profile_encoder(
            profile_memory.transpose(0, 1)).unsqueeze(0)
        self.decoder.load_memory(input_batches.transpose(0, 1))

        # Prepare input and output variables
        decoder_input = Variable(torch.LongTensor([2] * batch_size))

        decoded_words = []
        all_decoder_outputs_vocab = Variable(torch.zeros(max(target_lengths), batch_size, self.nwords))
        all_decoder_outputs_ptr = Variable(torch.zeros(max(target_lengths), batch_size, input_batches.size(0)))
        # all_decoder_outputs_gate = Variable(torch.zeros(self.max_r, batch_size))
        # Move new Variables to CUDA

        if self.use_cuda:
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
            if self.use_cuda: decoder_input = decoder_input.cuda()

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

        loss_v = self.cross_entropy(all_decoder_outputs_vocab.contiguous().view(-1, self.nwords),
                                    target_batches.contiguous().view(-1))
        loss_ptr = self.cross_entropy(all_decoder_outputs_ptr.contiguous().view(-1, input_batches.size(0)),
                                      target_index.contiguous().view(-1))

        if self.loss_weighting:
            loss = loss_ptr/(2*self.loss_weights[0]*self.loss_weights[0]) + loss_v/(2*self.loss_weights[1]*self.loss_weights[1]) + \
               torch.log(self.loss_weights[0] * self.loss_weights[1])
        else:
            loss = loss_ptr + loss_v

        self.loss += loss.item()
        self.vloss += loss_v.item()
        self.ploss += loss_ptr.item()
        self.n += 1

        # Set back to training mode
        self.encoder.train(True)
        self.decoder.train(True)
        self.profile_encoder.train(True)
        return decoded_words, self.from_whichs  # , acc_ptr, acc_vac

    def save_models(self, path):
        torch.save(self.encoder.state_dict(), os.path.join(path, 'encoder.pth'))
        torch.save(self.decoder.state_dict(), os.path.join(path, 'decoder.pth'))
        torch.save(self.profile_encoder.state_dict(), os.path.join(path, 'profile_encoder.pth'))

    def load_models(self, path: str = '.'):
        self.encoder.load_state_dict(torch.load(os.path.join(path, 'encoder.pth')))
        self.profile_encoder.load_state_dict(torch.load(os.path.join(path, 'profile_encoder.pth')))
        self.decoder.load_state_dict(torch.load(os.path.join(path, 'decoder.pth')))

