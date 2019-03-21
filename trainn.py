import numpy as np
import torch

from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
import pdb 

from model import Model
from data import find_entities, read_dataset, TextDataset, collate_fn


args = {
	'hops': 3,
	'emb_size': 128,
	'gru_size': 128,
	'batch_size': 8,
}


# Read in the data
kb_entries = find_entities("data/dialog-bAbI-tasks/dialog-babi-kb-all.txt")
train, w2i = list(read_dataset("data/dialog-bAbI-tasks/dialog-babi-task5-full-dialogs-trn.txt", kb_entries))
data = TextDataset(train, w2i)
batch_size = 8
data_loader = torch.utils.data.DataLoader(dataset=data,
                                              batch_size=batch_size,
                                              shuffle=True,
                                              collate_fn=collate_fn)
# print(w2i)
model = Model(3, len(w2i)+1, args['emb_size'], args['gru_size'], args['batch_size'], w2i)

use_cuda = torch.cuda.is_available()

if use_cuda:
    # TYPE = torch.cuda.LongTensor
    model.cuda()

# dec = Decoder()

for batch in data_loader:
    model.train(batch[0], batch[1], batch[2], batch[3])



# w2i = defaultdict(lambda: UNK, w2i)
# dev = list(read_dataset("topicclass_valid.txt"))
# test = list(read_test("topicclass_test.txt"))
nwords = len(w2i)
ntags = len(t2i)