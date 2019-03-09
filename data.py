import numpy as np
import torch

from collections import 
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


class CNNclass(torch.nn.Module):
    def __init__(self, nwords, emb_size, num_filters, ntags, embed_matrix, filters):
        super(CNNclass, self).__init__()

        self.emb_size = emb_size
        """ layers """
        self.embedding = torch.nn.Embedding(nwords, emb_size, padding_idx=0)
        self.embedding.weight.data.copy_(torch.from_numpy(embed_matrix))
        self.convs = torch.nn.ModuleList([torch.nn.Conv2d(1, 50, (f, emb_size)) for f in filters])
        self.bns = torch.nn.BatchNorm2d(50)

        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(0.5)

        self.projection_layer = torch.nn.Linear(in_features=len(filters) * 50, out_features=ntags, bias=True)
        # Initializing the projection layer
        torch.nn.init.xavier_uniform_(self.projection_layer.weight)

    def forward(self, words, mask=None):
        emb = self.embedding(words)
        if mask is not None:
            emb = emb * mask.type(torch.cuda.FloatTensor).unsqueeze(2).expand(emb.size())
            # emb = emb.div(emb.sum(-1, keepdim=True) * mask.type(torch.cuda.FloatTensor))
        emb = emb.unsqueeze(1)

        h = [self.relu(c(emb)) for c in self.convs]
        h = [self.bns(x) for x in h]
        h = [F.max_pool1d(i.squeeze(3), i.size(2)).squeeze(2) for i in h]
        h = torch.cat(h, 1)
        h = self.dropout(h)
        h = self.projection_layer(h)
        return h


word2vec = KeyedVectors.load_word2vec_format("GoogleNews-vectors-negative300.bin", binary=True)

# Functions to read in the corpus
w2i = defaultdict(lambda: len(w2i))
t2i = defaultdict(lambda: len(t2i))
PAD = w2i["<pad>"]
UNK = w2i["<unk>"]


# Filters (out channels, in_channels)

def read_dataset(string):
    bytez = open(string, 'rb').read()
    bytez = str(bytez, 'utf-8')
    for line in bytez.splitlines():
        tag, words = line.lower().strip().split(" ||| ")
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

# https://stackoverflow.com/questions/49710537/pytorch-gensim-how-to-load-pre-trained-word-embeddings
embed_matrix = []
for word in w2i.keys():
    if word in word2vec.vocab:
        embed_matrix.append(word2vec.word_vec(word))
    else:
        embed_matrix.append(np.random.uniform(-0.25, 0.25, 300))
embed_matrix = np.array(embed_matrix)