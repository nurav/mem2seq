import numpy as np
import torch

from tqdm import tqdm
import argparse
import datetime


from data import find_entities, read_dataset, TextDataset, collate_fn

from model_test import *

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default='dialog-babi-task1-API-calls')
parser.add_argument("--log", action='store_true', default=True)
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--val", type=int, default=5)
parser.add_argument("--name", type=str, default='task5')
parser.add_argument("-b", type=int, default=8, dest='batch_size')
parser.add_argument("--cuda", action='store_true', default=True)

args = parser.parse_args()

kb_entries = find_entities("data/dialog-bAbI-tasks/dialog-babi-kb-all.txt")
train, w2i = list(read_dataset(f"data/dialog-bAbI-tasks/{args.task}-trn.txt", kb_entries))
data = TextDataset(train, w2i)
batch_size = 8
train = torch.utils.data.DataLoader(dataset=data,
                                              batch_size=args.batch_size,
                                              shuffle=True,
                                              collate_fn=collate_fn)
# pdb.set_trace()
model = Model(3, len(w2i), 128, 128, w2i)


with open(f"log-{str(datetime.datetime.now())}-{args.name}", 'w') as log_file:
    for epoch in range(300):
        pbar = tqdm(enumerate(train), total=len(train))
        for i, batch in pbar:
            model.train_batch(batch[0].transpose(0,1), batch[1].transpose(0,1), batch[2].transpose(0,1), batch[3].transpose(0,1), i==0, batch[4], batch[5],  8)
            pbar.set_description(model.print_loss())
            if args.log:
                print(f"epoch {epoch}: {model.print_loss()}", file=log_file)

        if epoch % args.val == 0:
            import os
            os.makedirs('checkpoints/ckpt-'+str(epoch), exist_ok=True)
            model.save_models('checkpoints/ckpt-'+epoch)



