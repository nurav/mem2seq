import numpy as np
import torch

from collections import defaultdict
from torch.utils.data import Dataset, DataLoader

use_cuda = torch.cuda.is_available()

class TextDataset(Dataset):
    def __init__(self, memory, w2i, ):
        self.memory = memory
        self.w2i = w2i
        self.preprocess()

    def preprocess(self):
        """ performs word to index conversion for every element
                """
        for idx in range(len(self.memory)):
            # pdb.set_trace()
            m = self.memory[idx]
            context_seq = m[0]
            bot_seq = m[1]
            index_seq = m[2]
            gate_seq = m[3]
            profile_mem = m[5]
            # print(len(context_seq), len(bot_seq), len(index_seq), len(gate_seq))
            # print(m)
            new_context_seq = []
            new_prof_mem = []
            for c in context_seq:
                l = []
                for word in c:
                    l.append(self.w2i[word])
                new_context_seq.append(l)

            for p in profile_mem:
                l2 = []
                for word in p:
                    l2.append(self.w2i[word])
                new_prof_mem.append(l2)

            new_bot_seq = []
            for word in bot_seq.split(' '):
                new_bot_seq.append(self.w2i[word])
            new_bot_seq.append(self.w2i['<eos>'])
            index_seq.append(len(context_seq) - 1)
            gate_seq.append(False)
            m.append(new_context_seq)
            m.append(new_bot_seq)
            m.append(new_prof_mem)

    def __len__(self):
        return len(self.memory)

    def __getitem__(self, idx):
        return self.memory[idx][6], self.memory[idx][7], self.memory[idx][2], self.memory[idx][3], self.memory[idx][4], self.memory[idx][0], self.memory[idx][1], self.memory[idx][8]


def collate_fn(batch):
    """ pads the sequences to form tensors
    """
    batch.sort(key=lambda x: -len(x[0]))
    context_lengths = [len(x[0]) for x in batch]
    target_lengths = [len(x[1]) for x in batch]

    max_len_context = len(batch[0][0])
    max_len_target = max([len(x[1]) for x in batch])
    profile_len = len(batch[0][7])

    context = [np.array(x[0]) for x in batch]
    target = [np.array(x[1]) for x in batch]
    index = [np.array(x[2]) for x in batch]
    gate = [np.array(x[3]) for x in batch]
    dialog_idxs = [np.array(x[4]) for x in batch]
    context_words = [x[5] for x in batch]
    target_words = [x[6] for x in batch]
    profile_memory = [np.array(x[7]) for x in batch]

    out_context = np.zeros((len(batch), max_len_context, 3), dtype=int)
    out_target = np.zeros((len(batch), max_len_target), dtype=np.int64)
    out_index = np.zeros((len(batch), max_len_target), dtype=np.int64)
    out_gate = np.zeros((len(batch), max_len_target))
    out_dialog_idxs = np.zeros((len(batch)), dtype=np.int64)
    out_prof_mem = np.zeros((len(batch), profile_len, 3), dtype=np.int64)

    for i, x in enumerate(batch):
        out_context[i, 0:len(batch[i][0]), :] = context[i]
        out_target[i, 0:len(batch[i][1])] = target[i]
        out_index[i, 0:len(batch[i][2])] = index[i]
        out_gate[i, 0:len(batch[i][3])] = gate[i]
        out_dialog_idxs[i] = dialog_idxs[i]
        out_prof_mem[i, 0:len(batch[i][7]), :] = profile_memory[i]

    if use_cuda:
        return torch.from_numpy(out_context).cuda(), torch.from_numpy(out_target).cuda(), torch.from_numpy(out_index).cuda(), \
           torch.from_numpy(out_gate).cuda(), context_lengths, target_lengths, torch.from_numpy(out_dialog_idxs).cuda(), context_words, target_words, torch.from_numpy(out_prof_mem).cuda()
    else:
        return torch.from_numpy(out_context), torch.from_numpy(out_target), torch.from_numpy(out_index), torch.from_numpy(
            out_gate), context_lengths, target_lengths, out_dialog_idxs, context_words, target_words, torch.from_numpy(out_prof_mem)


def find_entities(filename):
    """ input: .txt file containing all kb entries
        output: a list containing all the kb entities (last element of each line in the kb entries)
    """
    bytez = open(filename, 'rb').read()
    bytez = str(bytez, 'utf-8')
    kb_entries = set()
    for line in bytez.splitlines():
        if '\t' in line:
            kb_entries.add(line.split('\t')[1])
    return (list(kb_entries))


# Filters (out channels, in_channels)

def read_dataset(string, kb_entries):
    """ input: .txt file containing dialogues
        output: a list with elements: context for every line, bot responses, index sequence, gate sequence
        Context: It is a list of lists, each list corresponds to a word in the context and is of the form [word, '$s', t0]
        bot response: It is a list of all the bot utterances
        index sequence: it is a list whose elements are the last occurrence of a word in a bot response in the context
        gate sequence: it is a list whose element is 1 if a word in bot response is present in context and 0 if absent """
    bytez = open(string, 'rb').read()
    bytez = str(bytez, 'utf-8')
    memory = []
    context = []
    profile_memory = []
    time = 1
    w2i = defaultdict(lambda: len(w2i))
    t2i = defaultdict(lambda: len(t2i))
    PAD = w2i["<pad>"]  # 0
    UNK = w2i["<unk>"]  # 1
    EOS = w2i["<eos>"]  # 2
    SOS = w2i["<sos>"]  # 3
    dialog_idx = 0
    _ = w2i['$u']
    _ = w2i['$s']

    for line in bytez.splitlines():
        if len(line) == 0:
            context = []
            time = 1
            sentinel = []
            idx = []
            dialog_idx += 1


        elif '\t' not in line:
            if line.split(' ')[0] == '1':
                temp = [[word, '$u', 't' + str(time)] for word in line.split(' ')[1:]]
                profile_memory.append(temp)
                for w in temp:
                    for z in w:
                        _ = w2i[z]

            else:
                nid, sent = line.split(' ', 1)
                sent_new = []
                sent_token = sent.split(' ')

                if sent_token[1] == "R_rating":
                    sent_token_2 = sent_token
                    sent_token_2.extend(["<pad>"] * (3 - len(sent_token)))
                else:
                    sent_token_2 = sent_token[::-1]
                    sent_token_2.extend(["<pad>"] * (3 - len(sent_token)))
                sent_new.append(sent_token_2)
                context.extend(sent_new)

                for s in sent_new:
                    for w in s:
                        _ = w2i[w]

        else:
            sentinel = []  # gate
            idx = []  # index

            user, bot = line.split('\t')
            user = user.split(' ')[1:]  # list of words in user utterance

            context.extend([[word, '$u', 't' + str(time)] for word in user])

            _ = w2i['t' + str(time)]
            for w in user:
                _ = w2i[w]

            for word in bot.split(' '):
                index = [i for i, w in enumerate(context) if w[0] == word]
                # print(index)
                idx.append(max(index) if index else len(context))
                sentinel.append(bool(index))

            context_new = context.copy()

            context_new.extend([['$$$$'] * 3])

            memory.append([context_new, bot, idx, sentinel, dialog_idx, profile_memory[-1]])  ##### final output

            context.extend([[word, '$s', 't' + str(time)] for word in bot.split(' ')])

            for w in bot.split(' '):
                _ = w2i[w]

            _ = w2i['t' + str(time)]
            time += 1
    # pdb.set_trace()
    return memory, w2i


