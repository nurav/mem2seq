import os.path
import torch
from tqdm import tqdm
import logging
import pickle

import numpy as np
import datetime

class ExperimentRunnerBase(torch.nn.Module):

    def __init__(self, args):
        super(ExperimentRunnerBase, self).__init__()
        data = args.data
        self.name = args.name
        self.task = args.task
        self.batch_size = args.batch_size
        self.epochs = args.epochs
        self.loss_weighting = args.loss_weighting
        self.TYPE = torch.LongTensor
        self.TYPEF = torch.FloatTensor
        self.use_cuda = torch.cuda.is_available()
        self.out_file = args.out_file
        self.from_which_enable = args.from_which
        self.small_data = args.small

        if self.use_cuda:
            self.TYPE = torch.cuda.LongTensor
            self.TYPEF = torch.cuda.FloatTensor

        if data == "babi":
            from data_babi import find_entities, read_dataset, TextDataset, collate_fn
            data_dir = "data/dialog-bAbI-tasks"
            kb_path = "dialog-babi-kb-all.txt"
            data_file_prefix = self.getBabiDataNames(self.task)
        elif data == "personal_context":
            from data_personal_context import find_entities, read_dataset, TextDataset, collate_fn

        elif data == "personal":
            from data_personalized import find_entities, read_dataset, TextDataset, collate_fn
        else:
            raise ModuleNotFoundError()

        if data.startswith("personal"):
            data_dir = "data/personalized-dialog-dataset/full"
            if self.small_data:
                data_dir = "data/personalized-dialog-dataset/small"
            kb_path = "personalized-dialog-kb-all.txt"
            data_file_prefix = self.getPersonalDataNames(args.task)

        self.kb_entries = find_entities(os.path.join(data_dir, "../full", kb_path))
        train, self.w2i = list(read_dataset(os.path.join(data_dir, f"{data_file_prefix}-trn.txt"), self.kb_entries))
        dev, _ = list(read_dataset(os.path.join(data_dir, f"{data_file_prefix}-dev.txt"), self.kb_entries))
        test, _ = list(read_dataset(os.path.join(data_dir, f"{data_file_prefix}-tst.txt"), self.kb_entries))

        self.data_train = TextDataset(train, self.w2i)
        self.data_dev = TextDataset(dev, self.w2i)
        self.data_test = TextDataset(test, self.w2i)
        self.i2w = {v: k for k, v in self.w2i.items()}

        self.train_data_loader = torch.utils.data.DataLoader(dataset=self.data_train,
                                                             batch_size=self.batch_size,
                                                             shuffle=True,
                                                             collate_fn=collate_fn,
                                                             pin_memory=True,
                                                             num_workers=5)

        self.dev_data_loader = torch.utils.data.DataLoader(dataset=self.data_dev,
                                                           batch_size=self.batch_size,
                                                           shuffle=False,
                                                           collate_fn=collate_fn,
                                                           pin_memory=True,
                                                           num_workers=5)

        self.test_data_loader = torch.utils.data.DataLoader(dataset=self.data_test,
                                                            batch_size=self.batch_size,
                                                            shuffle=False,
                                                            collate_fn=collate_fn,
                                                            pin_memory=True,
                                                            num_workers=5)


        self.args = args
        self.nwords = len(self.w2i)
        self.cross_entropy = torch.nn.CrossEntropyLoss()
        self.n = 1
        self.loss = 0
        self.ploss = 0
        self.vloss = 0
        self.acc = 0
        self.avg_best = 0

        self.train_plot_data = {
            'train': {
                'batch': [],
                'epoch': [],
                'loss': [],
                'vocab_loss': [],
                'ptr_loss': [],
            }}
        self.val_plot_data ={
            'val': {
                'batch': [],
                'loss': [],
                'vocab_loss': [],
                'ptr_loss': [],
                'wer': [],
                'acc': []
            }}
        self.test_plot_data = {
            'test': {
                'loss': [],
                'vocab_loss': [],
                'ptr_loss': [],
            }}
        if self.loss_weighting:
            self.loss_weights = torch.tensor([1.0, 1.0], requires_grad=True)
            if self.use_cuda:
                self.loss_weights = self.loss_weights.cuda()
            self.loss_weights = torch.nn.Parameter(self.loss_weights)

    def trainer(self):
        with open(f"log-{str(datetime.datetime.now())}-{self.name}", 'w') as log_file:
            try:
                for epoch in range(self.epochs):
                    pbar = tqdm(enumerate(self.train_data_loader), total=len(self.train_data_loader))
                    for i, batch in pbar:
                        if self.use_cuda:
                            batch[0] = batch[0].to('cuda', non_blocking=True)
                            batch[1] = batch[1].to('cuda', non_blocking=True)
                            batch[2] = batch[2].to('cuda', non_blocking=True)
                            batch[3] = batch[3].to('cuda', non_blocking=True)
                            batch[4] = batch[4].to('cuda', non_blocking=True)
                            if self.__class__.__name__.startswith("Split"):
                                batch[10] = batch[10].to('cuda', non_blocking=True)
                        self.train()
                        loss, vloss, ploss = self.train_batch_wrapper(batch, i == 0, 8)
                        pbar.set_description(self.print_loss())
                        if self.args.log:
                            print(f"epoch {epoch}: {self.print_loss()}", file=log_file)
                        self.train_plot_data['train']['batch'].append(i)
                        self.train_plot_data['train']['epoch'].append(epoch)
                        self.train_plot_data['train']['loss'].append(loss)
                        self.train_plot_data['train']['vocab_loss'].append(vloss)
                        self.train_plot_data['train']['ptr_loss'].append(ploss)
                    if epoch % self.args.val == 0:
                        os.makedirs('checkpoints/ckpt-' + str(self.name) + '-' + str(epoch), exist_ok=True)
                        self.save_models('checkpoints/ckpt-' + str(self.name) + '-' + str(epoch))
                        self.eval()
                        self.acc = self.evaluate(self.dev_data_loader, self.avg_best, self.kb_entries, self.i2w, epoch)
                        self.scheduler.step(self.acc)
                    # if 'Mem2Seq' in args['decoder']:
                    #     model.scheduler.step(acc)
                    if self.acc is None or self.avg_best is None:
                        continue

                    if (self.acc >= self.avg_best):
                        self.avg_best = self.acc
                        cnt = 0
                    else:
                        cnt += 1
                    if cnt == 5: break
                    if self.acc == 1.0: break
            except KeyboardInterrupt as e:
                out_file = open(f"plot-data-{self.name}.pkl", 'wb')
                out_file.write(pickle.dumps((self.train_plot_data, self.val_plot_data)))
                out_file.close()
                print(e)
                exit(0)
        out_file = open(f"plot-data-{self.name}.pkl", 'wb')
        out_file.write(pickle.dumps((self.train_plot_data, self.val_plot_data)))
        out_file.close()

    def getPersonalDataNames(self, tasknum):
        mapping = {"1": "personalized-dialog-task1-API-calls",
                   "2": "personalized-dialog-task2-API-refine",
                   "3": "personalized-dialog-task3-options",
                   "4": "personalized-dialog-task4-info",
                   "5": "personalized-dialog-task5-full-dialogs"}
        return mapping[str(tasknum)]

    def getBabiDataNames(self, tasknum):
        mapping = {"1" : "dialog-babi-task1-API-calls",
                   "2" : "dialog-babi-task2-API-refine",
                   "3" : "dialog-babi-task3-options",
                   "4" : "dialog-babi-task4-phone-address",
                   "5" : "dialog-babi-task5-full-dialogs",
                   "6" : "dialog-babi-task6-dstc2"}
        return mapping[str(tasknum)]

    def losses(self):
        return self.loss / self.n, self.ploss / self.n, self.vloss / self.n

    def _optimize(self, predicted_answers, true_answers):
        """Implement this in subclass"""
        raise NotImplementedError()

    def print_loss(self):
        print_loss_avg = self.loss / self.n
        print_ploss = self.ploss / self.n
        print_vloss = self.vloss / self.n
        self.n += 1
        return 'L:{:.5f}, VL:{:.5f}, PL:{:.5f}, LW: {:.3f} {:.3f}'.format(print_loss_avg, print_vloss, print_ploss, self.loss_weights[0], self.loss_weights[1])

    def evaluate(self, dev, avg_best, kb_entries, i2w, epoch):
        self.loss = 0
        self.ploss = 0
        self.vloss = 0
        self.n = 1

        self.incorrect_sentinel = 0
        self.i2w = i2w

        def wer(r, h):
            """
            This is a function that calculate the word error rate in ASR.
            You can use it like this: wer("what is it".split(), "what is".split())
            """
            # build the matrix
            d = np.zeros((len(r) + 1) * (len(h) + 1), dtype=np.uint8).reshape((len(r) + 1, len(h) + 1))
            for i in range(len(r) + 1):
                for j in range(len(h) + 1):
                    if i == 0:
                        d[0][j] = j
                    elif j == 0:
                        d[i][0] = i
            for i in range(1, len(r) + 1):
                for j in range(1, len(h) + 1):
                    if r[i - 1] == h[j - 1]:
                        d[i][j] = d[i - 1][j - 1]
                    else:
                        substitute = d[i - 1][j - 1] + 1
                        insert = d[i][j - 1] + 1
                        delete = d[i - 1][j] + 1
                        d[i][j] = min(substitute, insert, delete)
            result = float(d[len(r)][len(h)]) / len(r) * 100
            # result = str("%.2f" % result) + "%"
            return result

        logging.info("STARTING EVALUATION")
        acc_avg = 0.0
        wer_avg = 0.0
        bleu_avg = 0.0
        acc_P = 0.0
        acc_V = 0.0
        ref = []
        hyp = []
        ref_s = ""
        hyp_s = ""
        dialog_acc_dict = {}

        global_entity_list = kb_entries

        pbar = tqdm(enumerate(dev), total=len(dev))
        for j, data_dev in pbar:
            profile_mem = None
            if self.__class__.__name__.startswith("Split"):
                profile_mem = data_dev[9]
            if self.use_cuda:
                data_dev[0] = data_dev[0].to('cuda', non_blocking=True)
                data_dev[1] = data_dev[1].to('cuda', non_blocking=True)
                data_dev[2] = data_dev[2].to('cuda', non_blocking=True)
                data_dev[3] = data_dev[3].to('cuda', non_blocking=True)
                if self.__class__.__name__.startswith("Split"):
                    profile_mem = profile_mem.to('cuda', non_blocking=True)
            if profile_mem != None:
                profile_mem = profile_mem.transpose(0,1)
            words, from_whichs = self.evaluate_batch(len(data_dev[1]), data_dev[0].transpose(0, 1), data_dev[4],
                                        data_dev[1].transpose(0, 1), data_dev[5],
                                        data_dev[2].transpose(0, 1), data_dev[3].transpose(0, 1), data_dev[7],
                                        profile_mem)

            transposed_words = [[row[i] for row in words] for i in range(len(words[0]))]
            if self.from_which_enable:
                transposed_fromwhich = [[row[i] for row in from_whichs] for i in range(len(from_whichs[0]))]

            self.out_file = self.name if self.out_file == '' else self.out_file
            with open(self.out_file, 'a') as f:
                if j==0:
                    f.write("Epoch {}\n".format(epoch))
                f.write('------------truth---------------\n\n')
                [f.write(w + '\n') for w in data_dev[8]]

                f.write('------------response-------------\n\n')
                [f.write(' '.join(w) + '\n') for w in transposed_words]
                f.write('\n')
                if self.from_which_enable:
                    [f.write(" ".join(w) + '\n') for w in transposed_fromwhich]
                    f.write("\n")
            acc = 0
            w = 0
            temp_gen = []

            for i, row in enumerate(np.transpose(words)):
                st = ''
                for e in row:
                    if e == '<eos>':
                        break
                    else:
                        st += e + ' '
                temp_gen.append(st)
                correct = data_dev[8][i]
                ### compute F1 SCORE
                st = st.lstrip().rstrip()
                correct = correct.lstrip().rstrip()

                if data_dev[6][i] not in dialog_acc_dict.keys():
                    dialog_acc_dict[data_dev[6][i].item()] = []
                if (correct == st):
                    acc += 1
                    dialog_acc_dict[data_dev[6][i].item()].append(1)
                else:
                    dialog_acc_dict[data_dev[6][i].item()].append(0)

                w += wer(correct, st)
                ref.append(str(correct))
                hyp.append(str(st))
                ref_s += str(correct) + "\n"
                hyp_s += str(st) + "\n"

            acc_avg += acc / float(len(data_dev[1]))
            wer_avg += w / float(len(data_dev[1]))
            pbar.set_description("R:{:.4f},W:{:.4f},I:{:.4f}".format(acc_avg / float(len(dev)),
                                                                     wer_avg / float(len(dev)),
                                                                     self.incorrect_sentinel / float(len(dev))))
            self.val_plot_data['val']['batch'].append(j)
            self.val_plot_data['val']['acc'].append(acc_avg / float(len(dev)))
            self.val_plot_data['val']['wer'].append(wer_avg / float(len(dev)))

            self.val_plot_data['val']['loss'].append(self.losses()[0])
            self.val_plot_data['val']['vocab_loss'].append(self.losses()[1])
            self.val_plot_data['val']['ptr_loss'].append(self.losses()[2])

        # dialog accuracy

        dia_acc = 0
        for k in dialog_acc_dict.keys():
            if len(dialog_acc_dict[k]) == sum(dialog_acc_dict[k]):
                dia_acc += 1
        logging.info("Dialog Accuracy:\t" + str(dia_acc * 1.0 / len(dialog_acc_dict.keys())))

        acc_avg = acc_avg / float(len(dev))

        if (acc_avg >= avg_best):
            return acc_avg
        return avg_best
