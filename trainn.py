import numpy as np
import torch

from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
import pdb 
from tqdm import tqdm
from model_test import Model
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
dev, _ = list(read_dataset("data/dialog-bAbI-tasks/dialog-babi-task5-full-dialogs-dev.txt", kb_entries))
# train, w2i = list(read_dataset("data/dialog-bAbI-tasks/sample.txt", kb_entries))
data_train = TextDataset(train, w2i)
data_dev = TextDataset(dev, w2i)

batch_size = 8
train_data_loader = torch.utils.data.DataLoader(dataset=data_train,
                                              batch_size=batch_size,
                                              shuffle=True,
                                              collate_fn=collate_fn)

dev_data_loader = torch.utils.data.DataLoader(dataset=data_dev,
                                              batch_size=batch_size,
                                              shuffle=True,
                                              collate_fn=collate_fn)

# print(w2i)
model = Model(3, len(w2i), args['emb_size'], args['gru_size'], args['batch_size'], w2i)

use_cuda = torch.cuda.is_available()

if use_cuda:
    # TYPE = torch.cuda.LongTensor
    model.cuda()

# dec = Decoder()


for epoch in range(0,100): # (TODO): Change this
    pbar = tqdm(enumerate(train_data_loader), total=len(train_data_loader))
    for i, batch in pbar:
        if i == 100:
            abcd = 0
        # 0 =context , 1=target, 2=index, 3= gate, 4=context lengths, 5=target lengths
        model.trainer(batch[0], batch[1], batch[2], batch[3], i==0, batch[4], batch[5], 8) #(TODO): Fix gate sequence
        pbar.set_description(model.show_loss())
    for j, batch_dev in enumerate(dev_data_loader):
        model.eval()
        response_acc = model.evaluate(batch_dev[0], batch[1])
        print("Per response accuracy ", response_acc)
        model.train()



# for batch in data_loader:
#     model.train(batch[0], batch[1], batch[2], batch[3])



# w2i = defaultdict(lambda: UNK, w2i)
# dev = list(read_dataset("topicclass_valid.txt"))
# test = list(read_test("topicclass_test.txt"))
nwords = len(w2i)
ntags = len(t2i)