# import numpy as np
# import torch

# from tqdm import tqdm

from data import find_entities, read_dataset, TextDataset, collate_fn

from model_test import *

BLEU = False


kb_entries = find_entities("data/dialog-bAbI-tasks/dialog-babi-kb-all.txt")
train, w2i = list(read_dataset("data/dialog-bAbI-tasks/dialog-babi-task5-full-dialogs-trn.txt", kb_entries))
data = TextDataset(train, w2i)
batch_size = 8
train = torch.utils.data.DataLoader(dataset=data,
                                              batch_size=8,
                                              shuffle=True,
                                              collate_fn=collate_fn)
# pdb.set_trace()
model = Model(3, len(w2i), 128, 128, w2i)

for epoch in range(300):
    pbar = tqdm(enumerate(train), total=len(train))
    for i, batch in pbar:
        model.train_batch(batch[0].transpose(0,1), batch[1].transpose(0,1), batch[2].transpose(0,1), batch[3].transpose(0,1), i==0, batch[4], batch[5],  8)
        pbar.set_description(model.print_loss())



