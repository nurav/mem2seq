from matplotlib import pyplot as plt
import argparse
import pickle


parser = argparse.ArgumentParser()
parser.add_argument('--file', type=str, default='.')
args = parser.parse_args()

f = open(args.file, 'rb')
train, val = pickle.load(f)

print("reached here")

fig, pl = plt.subplots()
pl.plot(train['train']['loss'], label="Combined loss")
pl.plot(train['train']['vocab_loss'], label="Vocab loss")
pl.plot(train['train']['ptr_loss'], label="Pointer loss")

handles, labels = pl.get_legend_handles_labels()
pl.legend(handles, labels)

pl.set(xlabel="Batch no.", ylabel="Losses", title="Training Losses vs Batch number (batch size 8)")
plt.yscale("log")

fig.savefig(f"{args.file.split('.')[:-1][0]}-plot.png")
val['val']['acc'] = [v * 100 for i, v in enumerate(val['val']['acc']) if i % 751 == 1]


fig, pl = plt.subplots()
# pl.plot(val['val']['loss'], label="Combined loss")
# pl.plot(val['val']['vocab_loss'], label="Vocab loss")
# pl.plot(val['val']['ptr_loss'], label="Pointer loss")
pl.plot(val['val']['acc'], label="Accuracy")

handles, labels = pl.get_legend_handles_labels()
pl.legend(handles, labels)

pl.set(xlabel="Epoch", ylabel="Losses", title="Validation Losses vs Batch number (batch size 8)")

fig.savefig(f"{args.file.split('.')[:-1][0]}-plot-val.png")