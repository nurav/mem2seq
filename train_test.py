import numpy as np
# import torch
#
# from collections import defaultdict
# from torch.utils.data import Dataset, DataLoader
# import pdb
# from tqdm import tqdm
# from model import Model
from data import find_entities, read_dataset, TextDataset, collate_fn
# from model_test import Mem2Seq
#
#
args = {
	'hops': 3,
	'emb_size': 128,
	'gru_size': 128,
	'batch_size': 8,
}


from model_test import *
# import pdb

BLEU = False


# Configure models
avg_best, cnt, acc = 0.0, 0, 0.0
cnt_1 = 0
### LOAD DATA
# train, dev, test, testOOV, lang, max_len, max_r = prepare_data_seq(args['task'], batch_size=int(args['batch']),
#                                                                    shuffle=False)
kb_entries = find_entities("data/dialog-bAbI-tasks/dialog-babi-kb-all.txt")
train, w2i = list(read_dataset("data/dialog-bAbI-tasks/dialog-babi-task5-full-dialogs-trn.txt", kb_entries))
dev, _ = list(read_dataset("data/dialog-bAbI-tasks/dialog-babi-task5-full-dialogs-dev.txt", kb_entries))
# train, w2i = list(read_dataset("data/dialog-bAbI-tasks/sample.txt", kb_entries))
data_train = TextDataset(train, w2i)
data_dev = TextDataset(dev, w2i)

batch_size = 8
train_data_loader = torch.utils.data.DataLoader(dataset=data_train,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              collate_fn=collate_fn)

dev_data_loader = torch.utils.data.DataLoader(dataset=data_dev,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              collate_fn=collate_fn)

model = Model(3, len(w2i), args['emb_size'], args['gru_size'], args['batch_size'], w2i)

use_cuda = torch.cuda.is_available()

if use_cuda:
    # TYPE = torch.cuda.LongTensor
    model.cuda()

# dec = Decoder()


for epoch in range(0,100): # (TODO): Change this
    pbar = tqdm(enumerate(train_data_loader), total=len(train_data_loader))
    for i, batch in pbar:
        # 0 =context , 1=target, 2=index, 3= gate, 4=context lengths, 5=target lengths
        model.train_batch(batch[0].transpose(0,1), batch[4], batch[1].transpose(0,1), batch[5], batch[2].transpose(0,1), batch[3].transpose(0,1), len(batch[0].transpose(0,1)), 8, 0.5, i==0) #(TODO): Fix gate sequence
        pbar.set_description(model.print_loss())
        break
    response_acc = []
    batch_sizes = []
    model.eval()
    for j, batch_dev in enumerate(dev_data_loader):
        response_acc.append(model.evaluate(batch_dev[0].transpose(0,1), batch_dev[1].transpose(0,1)))
        batch_sizes.append(batch_dev[1].shape[0])
    print("Per response accuracy ", np.average(response_acc, weights= batch_sizes))
    model.train()

        #pbar.set_description(model.print_loss())

    # if((epoch+1) % int(args['evalp']) == 0):
    #     acc = model.evaluate(dev,avg_best, BLEU)
    #     if 'Mem2Seq' in args['decoder']:
    #         model.scheduler.step(acc)
    #     if(acc >= avg_best):
    #         avg_best = acc
    #         cnt=0
    #     else:
    #         cnt+=1
    #     if(cnt == 5): break
    #     if(acc == 1.0): break


