import numpy as np
import torch

from tqdm import tqdm
import argparse
import datetime
import os
import pickle

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default='dialog-babi-task1-API-calls')
parser.add_argument("--model_personalized", action='store_true', default=False)
parser.add_argument("--data_personalized", action='store_true', default=False)
parser.add_argument("--log", action='store_true', default=True)
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--val", type=int, default=5)
parser.add_argument("--name", type=str, default='task4')
parser.add_argument("-b", type=int, default=8, dest='batch_size')
parser.add_argument("--cuda", action='store_true', default=False)
parser.add_argument("--load_from", type=str, default=None)
parser.add_argument("--test", action="store_true", default=False)
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--personalization_type", type=str, default='split_memory')

args = parser.parse_args()

kb_entries = None

avg_best = 0
acc = 0

global_memory = 0
global_memory_test = 0
global_memory_dev = 0
if args.model_personalized:
    if args.personalization_type == 'context':
        from model_personalized_context import *
    elif args.personalization_type == 'hidden':
        from model_personalized_hidden import *
    else:
        from model_personalized import *
else:
    from model import *
if args.data_personalized:
    if args.personalization_type == 'context':
        from data_personal_context import find_entities, read_dataset, TextDataset, collate_fn
    else:
        from data_specific_context import find_entities, read_dataset, TextDataset, collate_fn
    kb_entries = find_entities("data/personalized-dialog-dataset/full/personalized-dialog-kb-all.txt")
    train, w2i, global_memory = list(read_dataset(f"data/personalized-dialog-dataset/full/{args.task}-trn.txt", kb_entries))
    dev, _, global_memory_dev = list(read_dataset(f"data/personalized-dialog-dataset/full/{args.task}-dev.txt", kb_entries))
    test, _, global_memory_test = list(read_dataset(f"data/personalized-dialog-dataset/full/{args.task}-tst.txt", kb_entries))
else:
    from data import find_entities, read_dataset, TextDataset, collate_fn
    kb_entries = find_entities("data/dialog-bAbI-tasks/dialog-babi-kb-all.txt")
    train, w2i = list(read_dataset(f"data/dialog-bAbI-tasks/{args.task}-trn.txt", kb_entries))
    dev, _ = list(read_dataset(f"data/dialog-bAbI-tasks/{args.task}-dev.txt", kb_entries))
    test, _ = list(read_dataset(f"data/dialog-bAbI-tasks/{args.task}-tst.txt", kb_entries))

data_train = TextDataset(train, w2i, global_memory)
data_dev = TextDataset(dev, w2i, global_memory_dev)
data_test = TextDataset(test, w2i, global_memory_test)

i2w = {v: k for k, v in w2i.items()}

train_data_loader = torch.utils.data.DataLoader(dataset=data_train,
                                              batch_size=args.batch_size,
                                              shuffle=False,
                                              collate_fn=collate_fn,
                                                pin_memory=True,
                                                num_workers=2)

dev_data_loader = torch.utils.data.DataLoader(dataset=data_dev,
                                              batch_size=args.batch_size,
                                              shuffle=False,
                                              collate_fn=collate_fn,
                                              pin_memory=True,
                                              num_workers=2)

test_data_loader = torch.utils.data.DataLoader(dataset=data_test,
                                              batch_size=args.batch_size,
                                              shuffle=False,
                                              collate_fn=collate_fn,
                                               pin_memory=True,
                                               num_workers=2)
model = Model(3, len(w2i), 128, 128, w2i)

if args.cuda:
    model = model.cuda()

if args.load_from:
    model.load_models(args.load_from)

plot_data = {
    'train': {
        'batch': [],
        'epoch': [],
        'loss': [],
        'vocab_loss': [],
        'ptr_loss': [],
    },
    'val':{
        'batch': [],
        'loss': [],
        'vocab_loss': [],
        'ptr_loss': [],
        'wer': [],
        'acc': []
    },
    'test': {
        'loss': [],
        'vocab_loss': [],
        'ptr_loss': [],
    },
}

if args.test:
    model.eval()
    acc = model.evaluate(test_data_loader, 0, kb_entries, i2w)
    import sys; sys.exit(0)

with open(f"log-{str(datetime.datetime.now())}-{args.name}", 'w') as log_file:
    for epoch in range(args.epochs):
        pbar = tqdm(enumerate(train_data_loader), total=len(train_data_loader))
        for i, batch in pbar:
            model.train()
            if args.model_personalized:
                if args.personalization_type == 'split_memory' or args.personalization_type == 'hidden':
                    loss, vloss, ploss = model.train_batch(batch[0].transpose(0, 1), batch[1].transpose(0, 1), batch[2].transpose(0, 1),
                              batch[3].transpose(0, 1), i == 0, batch[4], batch[5], 8, batch[9].transpose(0,1), batch[10:20])
                else:
                    loss, vloss, ploss = model.train_batch(batch[0].transpose(0, 1), batch[1].transpose(0, 1), batch[2].transpose(0, 1),
                                      batch[3].transpose(0, 1), i == 0, batch[4], batch[5], 8)
            else:
                loss, vloss, ploss = model.train_batch(batch[0].transpose(0, 1), batch[1].transpose(0, 1), batch[2].transpose(0, 1),
                                  batch[3].transpose(0, 1), i == 0, batch[4], batch[5], 8)
            pbar.set_description(model.print_loss())


            if args.log:
                print(f"epoch {epoch}: {model.print_loss()}", file=log_file)

            plot_data['train']['batch'].append(i)
            plot_data['train']['epoch'].append(epoch)
            plot_data['train']['loss'].append(loss)
            plot_data['train']['vocab_loss'].append(vloss)
            plot_data['train']['ptr_loss'].append(ploss)

        if epoch % args.val == 0:
        #     response_acc = []
        #     batch_sizes = []
        #     model.eval()
        #     for j, batch_dev in enumerate(dev_data_loader):
        #         response_acc.append(model.evaluate(batch_dev[0].transpose(0, 1), batch_dev[1].transpose(0, 1)))
        #         batch_sizes.append(batch_dev[1].shape[0])
        #     print("Per response accuracy ", np.average(response_acc, weights=batch_sizes), file=log_file)
        #
        #     model.train()
        #
            os.makedirs('checkpoints/ckpt-' + str(args.name)+'-'+str(epoch), exist_ok=True)
            model.save_models('checkpoints/ckpt-' + str(args.name)+'-'+str(epoch))

        if (epoch % args.val == 0):
            model.eval()
            acc = model.evaluate(dev_data_loader,avg_best, kb_entries, i2w)
            model.scheduler.step(acc)
        # if 'Mem2Seq' in args['decoder']:
        #     model.scheduler.step(acc)
        if acc is None or avg_best is None:
            continue

        if(acc >= avg_best):
            avg_best = acc
            cnt=0
        else:
            cnt+=1
        if(cnt == 5): break
        if(acc == 1.0): break

out_file = open(f"plot-data-{args.name}.pkl", 'wb')
out_file.write(pickle.dumps((plot_data, model.plot_data)))
out_file.close()