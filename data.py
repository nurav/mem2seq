import numpy as np
import torch

from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
import pdb 



class TextDataset(Dataset):
    def __init__(self, memory, w2i):
        self.memory = memory
        self.w2i = w2i

    def __len__(self):
        return len(self.memory)

    def __getitem__(self, idx):
        # pdb.set_trace()
        context_seq = []
        bot_seq = []
        index_seq = []
        gate_seq = []

        m = self.memory[idx]
        context_seq = m[0]
        bot_seq = m[1]
        index_seq = m[2]
        gate_seq = m[3]

        new_context_seq = []
        for c in context_seq:
            l = []
            for word in c:
                l.append(self.w2i[word])
            new_context_seq.append(l)

        new_bot_seq = []
        for word in bot_seq.split(' '):
            new_bot_seq.append(self.w2i[word])
        new_bot_seq.append(EOS)
        index_seq.append(len(context_seq)-1)
        gate_seq.append(False)
        return new_context_seq, new_bot_seq, index_seq, gate_seq


def collate_fn(batch):
    # print(batch)
    batch.sort(key = lambda x: -len(x[0]))
    # print(batch)
    max_len_context = len(batch[0][0])
    max_len_target = max([len(x[1]) for x in batch])

    context = [np.array(x[0]) for x in batch]
    # pdb.set_trace()
    target = [np.array(x[1]) for x in batch]
    index = [np.array(x[2]) for x in batch]
    gate = [np.array(x[3]) for x in batch]

    out_context = np.zeros((len(batch), max_len_context, 3))
    out_target, out_index, out_gate = [np.zeros((len(batch), max_len_target))] * 3

    for i, x in enumerate(batch):
        out_context[i, 0:len(batch[i][0]), :] = context[i]
        out_target[i, 0:len(batch[i][1])] = target[i]
        out_index[i, 0:len(batch[i][2])] = index[i]
        out_gate[i, 0:len(batch[i][3])] = gate[i] 

    return torch.from_numpy(out_context), torch.from_numpy(out_target), torch.from_numpy(out_index), torch.from_numpy(out_gate)


# Functions to read in the corpus
w2i = defaultdict(lambda: len(w2i))
t2i = defaultdict(lambda: len(t2i))
PAD = w2i["<pad>"] #0
UNK = w2i["<unk>"] #1
EOS = w2i["<eos>"] #2


def find_entities(filename):
    bytez = open(filename, 'rb').read()
    bytez = str(bytez, 'utf-8')
    kb_entries = set()
    for line in bytez.splitlines():
        if '\t' not in line and len(line)>0:
            kb_entries.add(line.split(' ')[3])
    return(list(kb_entries))
# Filters (out channels, in_channels)

def read_dataset(string, kb_entries):
    bytez = open(string, 'rb').read()
    bytez = str(bytez, 'utf-8')
    memory = []
    context = []
    time = 1

    for line in bytez.splitlines():
        if len(line) == 0:
            context = []
            time = 1
            sentinel = []
            idx = []

            # dialog_idx = 0

        elif '\t' not in line:
            context.append([word for word in line.split(' ')[1:]])
            
            # for word in line.split(' '):
            #     context += word
               #w2i[word]
        else:
            sentinel = [] # gate
            idx = [] #index

            user, bot = line.split('\t')
            user = user.split(' ')[1:] # list of words in user utterance
            
            context.extend([[word,'$u','t'+str(time)] for word in user])
            
            for word in bot.split(' '):
                index = [i for i,w in enumerate(context) if w[0] == word]
                # print(index)
                idx.append(max(index) if index else len(context))
                sentinel.append(bool(index))

            context_new = context.copy()

            context_new.extend([['$$$$']*3])

            

            memory.append([context_new, bot, idx, sentinel]) ##### final output

            context.extend([[word,'$s','t'+str(time)] for word in bot.split(' ')])

            time += 1

        # tag, words = line.lower().strip().split("\t")
        # yield ([w2i[x] for x in words.split(" ")], t2i[tag])
    # print(memory)
    return memory



def read_test(string):
    bytez = open(string, 'rb').read()
    bytez = str(bytez, 'utf-8')
    for line in bytez.splitlines():
        _, words = line.lower().strip().split(" ||| ")
        yield [w2i[x] for x in words.split(" ")]


# Read in the data
kb_entries = find_entities("data/dialog-bAbi-tasks/dialog-babi-kb-all.txt")
train = list(read_dataset("data/dialog-bAbi-tasks/dialog-babi-task5trn.txt", kb_entries))
data = TextDataset(train, w2i)
# pdb.set_trace()
batch_size = 8
# print('here')
data_loader = torch.utils.data.DataLoader(dataset=data,
                                              batch_size=batch_size,
                                              shuffle=True,
                                              collate_fn=collate_fn)

for batch in data_loader:
    pdb.set_trace()

# w2i = defaultdict(lambda: UNK, w2i)
# dev = list(read_dataset("topicclass_valid.txt"))
# test = list(read_test("topicclass_test.txt"))
nwords = len(w2i)
ntags = len(t2i)


"""
Reads 
"""
def read_task(prefix):
    for set in ['dev', 'trn']:
        read_dataset()

def initialize_dict(path):
    bytez = open(path, 'rb').read()
    bytez = str(bytez, 'utf-8')
    for line in bytez.splitlines():
        if len(line) == 0:
            continue
        words = line.strip().split(" ")[1:]
        for word in words:
            w2i