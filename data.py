import numpy as np
import torch

from collections import defaultdict
from torch.utils.data import Dataset, DataLoader

class TextDataset(Dataset):
    def __init__(self, x):
        self.x = x

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx]


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
PAD = w2i["<pad>"]
UNK = w2i["<unk>"]



# Filters (out channels, in_channels)

def read_dataset(string):
    bytez = open(string, 'rb').read()
    bytez = str(bytez, 'utf-8')
    memory = []
    context = []

    for line in bytez.splitlines():
        if len(line) == 0:
            context = []
        elif '\t' not in line:
            for word in line.split(' '):
               memory += w2i[word]
            # put index of sum of word embeddings here
        else:

        tag, words = line.lower().strip().split("\t")
        yield ([w2i[x] for x in words.split(" ")], t2i[tag])


def read_test(string):
    bytez = open(string, 'rb').read()
    bytez = str(bytez, 'utf-8')
    for line in bytez.splitlines():
        _, words = line.lower().strip().split(" ||| ")
        yield [w2i[x] for x in words.split(" ")]


# Read in the data
train = list(read_dataset("topicclass_train.txt"))
w2i = defaultdict(lambda: UNK, w2i)
dev = list(read_dataset("topicclass_valid.txt"))
test = list(read_test("topicclass_test.txt"))
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

