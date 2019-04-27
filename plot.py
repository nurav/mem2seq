from matplotlib import pyplot as plt
import argparse
import pickle
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument('--file', type=str, default='.')
args = parser.parse_args()

f = open('Results/plot-data-task1_context.pkl', 'rb')
train, val = pickle.load(f)


epoch = train['train']['epoch']
batch = train['train']['batch']
loss = train['train']['loss']
vloss = train['train']['vocab_loss']
ploss = train['train']['ptr_loss']

batch_size = max(batch)
epoch = sorted(list(set(epoch)))
print(len(loss)/50, batch_size)
newloss=[]
newvloss=[]
newploss=[]
for i in range(batch_size,len(loss),batch_size+1):
    newloss.append(sum(loss[i-batch_size:i+1])/(batch_size+1))
    newvloss.append(sum(vloss[i-batch_size:i+1])/(batch_size+1))
    newploss.append(sum(ploss[i-batch_size:i+1])/(batch_size+1))
print(newloss)
# print(len(newloss))

newepoch = []
for i in range(0,max(epoch),5):
    newepoch.append(i)
fig, pl = plt.subplots()
pl.plot(newloss, label="Combined loss")
pl.plot(newvloss, label="Vocab loss")
pl.plot(newploss, label="Pointer loss")
plt.xticks(np.arange(0,len(newloss),5),newepoch)
plt.yscale('log')
handles, labels = pl.get_legend_handles_labels()
pl.legend(handles, labels)

pl.set(xlabel="Epochs", ylabel="Losses", title="Training Losses vs Epoch")

fig.savefig("task1-plot.png")

val_batch = val['val']['batch']
val_batch_size = max(val_batch)

val_acc = [v * 100 for i, v in enumerate(val['val']['acc']) if i % val_batch_size == 0]


fig, pl = plt.subplots()
# pl.plot(val['val']['loss'], label="Combined loss")
# pl.plot(val['val']['vocab_loss'], label="Vocab loss")
# pl.plot(val['val']['ptr_loss'], label="Pointer loss")
pl.plot(val_acc, label="Accuracy")
ticklist = []
tick = 0
for l in range(0,len(val_acc)-1):
    if l==0:
        tick = 0
        ticklist.append(tick)
    else:
        tick += 5
        ticklist.append(tick)
plt.xticks(np.arange(0,len(val_acc)),ticklist)

handles, labels = pl.get_legend_handles_labels()
pl.legend(handles, labels)

pl.set(xlabel="Epoch", ylabel="Accuracy", title="Validation Accuracy vs Epoch")

print(max(val_acc))
# fig.savefig(f"{args.file.split('.')[:-1][0]}-plot-val.png")