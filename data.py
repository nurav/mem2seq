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
        pdb.set_trace()
        context_seq = []
        bot_seq = []
        index_seq = []
        gate_seq = []
        # print("inside get item")
        # for m in memory[idx]:
        m = self.memory[idx]
        context_seq = m[0]
        bot_seq = m[1]
        index_seq = m[2]
        gate_seq = m[3]
        # print("appended stuff to sequence")
        
        # print(context_seq)
        new_context_seq = []
        for c in context_seq:
            # print(c)
            l = []
            for word in c:
                # print(word, w2i[word])
                l.append(self.w2i[word])
            new_context_seq.append(l)
        # print("here")

        new_bot_seq = []
        for word in bot_seq.split(' '):
            new_bot_seq.append(self.w2i[word])
        new_bot_seq.append(EOS)
            # print('there')
        index_seq.append(len(context_seq)-1)
        gate_seq.append(False)
        # print('hi')
        # print(new_context_seq)
        # print(new_bot_seq)
        # print(index_seq)
        # print(gate_seq)
        return new_context_seq, new_bot_seq, index_seq, gate_seq


def collate_fn(batch):
    max_len = max([len(x[0]) for x in batch])
    data = [np.array(x[0]) for x in batch]
    target = [np.array(x[1]) for x in batch]

    out = np.zeros((len(batch), max_len))
    out_target = np.zeros((len(batch)))
    mask = np.zeros((len(batch), max_len))
    for i, x in enumerate(batch):
        out[i, 0:len(batch[i][0])] = batch[i][0]
        mask[i, 0:len(batch[i][0])] = 1.0
        out_target[i] = batch[i][1]

    return torch.from_numpy(out), torch.from_numpy(out_target), torch.from_numpy(mask)


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
            context.extend([[word] for  word in line.split(' ')[1:]])
            
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