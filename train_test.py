# import numpy as np
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
# args = {
# 	'hops': 3,
# 	'emb_size': 128,
# 	'gru_size': 128,
# 	'batch_size': 8,
# }
#
#
# # Read in the data
# kb_entries = find_entities("data/dialog-bAbI-tasks/dialog-babi-kb-all.txt")
# train, w2i = list(read_dataset("data/dialog-bAbI-tasks/dialog-babi-task5-full-dialogs-trn.txt", kb_entries))
# # train, w2i = list(read_dataset("data/dialog-bAbI-tasks/sample.txt", kb_entries))
# data = TextDataset(train, w2i)
# batch_size = 8
# data_loader = torch.utils.data.DataLoader(dataset=data,
#                                               batch_size=8,
#                                               shuffle=True,
#                                               collate_fn=collate_fn)
# # print(w2i)
# model = Model(3, len(w2i), args['emb_size'], args['gru_size'], args['batch_size'], w2i)
# model_t = Mem2Seq(128, len(w2i), 0.001, 1, 0.2, False)
# use_cuda = torch.cuda.is_available()
#
# if use_cuda:
#     # TYPE = torch.cuda.LongTensor
#     model.cuda()
#
# # dec = Decoder()
#
#
# for epoch in range(0,100): # (TODO): Change this
#     pbar = tqdm(enumerate(data_loader), total=len(data_loader))
#     for i, batch in pbar:
#         # model.train(batch[0], batch[1], batch[2], batch[3], i==0, batch[4], batch[5]) #(TODO): Fix gate sequence
#         model_t.train_batch(batch[0].transpose(0,1), batch[4], batch[1].transpose(0,1), batch[5], batch[2].transpose(0,1), batch[3].transpose(0,1), 8, 0, 0.5, i==0)
#         pbar.set_description(model_t.print_loss())
#
# # for batch in data_loader:
# #     model.train(batch[0], batch[1], batch[2], batch[3])
#
#
#
# # w2i = defaultdict(lambda: UNK, w2i)
# # dev = list(read_dataset("topicclass_valid.txt"))
# # test = list(read_test("topicclass_test.txt"))
# nwords = len(w2i)
# # ntags = len(t2i)


# import numpy as np
# import logging
from tqdm import tqdm
#
# from utils.config import *
# from models.enc_vanilla import *
# from models.enc_Luong import *
# from models.enc_PTRUNK import *
from model_test import *
# import pdb

BLEU = False

# from utils.utils_babi import *

# Configure models
avg_best, cnt, acc = 0.0, 0, 0.0
cnt_1 = 0
### LOAD DATA
# train, dev, test, testOOV, lang, max_len, max_r = prepare_data_seq(args['task'], batch_size=int(args['batch']),
#                                                                    shuffle=False)
kb_entries = find_entities("data/dialog-bAbI-tasks/dialog-babi-kb-all.txt")
train, w2i = list(read_dataset("data/dialog-bAbI-tasks/dialog-babi-task5-full-dialogs-trn.txt", kb_entries))
# train, w2i = list(read_dataset("data/dialog-bAbI-tasks/sample.txt", kb_entries))
data = TextDataset(train, w2i)
batch_size = 8
train = torch.utils.data.DataLoader(dataset=data,
                                              batch_size=8,
                                              shuffle=False,
                                              collate_fn=collate_fn)
# pdb.set_trace()
model = Mem2Seq(128, len(w2i), 0.001, 1, 0.2, False)

for epoch in range(300):
    logging.info("Epoch:{}".format(epoch))
    # Run the train function
    pbar = tqdm(enumerate(train), total=len(train))
    # pdb.set_trace()
    for i, batch in pbar:
        # pdb.set_trace()
        model.train_batch(batch[0].transpose(0,1), batch[4], batch[1].transpose(0,1), batch[5], batch[2].transpose(0,1), batch[3].transpose(0,1), len(batch[0].transpose(0,1)), 8, 0.5, i==0)

        pbar.set_description(model.print_loss())

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


