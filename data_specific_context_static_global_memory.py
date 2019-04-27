import numpy as np
import torch

from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
import pdb
from model import Encoder

use_cuda = torch.cuda.is_available()


def create_global_memory(string, w2i):
    """ input: .txt file containing dialogues
        output: a list with elements: context for every line, bot responses, index sequence, gate sequence
        Context: It is a list of lists, each list corresponds to a word in the context and is of the form [word, '$s', t0]
        bot response: It is a list of all the bot utterances
        index sequence: it is a list whose elements are the last occurrence of a word in a bot response in the context
        gate sequence: it is a list whose element is 1 if a word in bot response is present in context and 0 if absent """
    bytez = open(string, 'rb').read()
    bytez = str(bytez, 'utf-8')
    memory = []
    context = []
    profile_memory = []
    time = 1
    # w2i = defaultdict(lambda: len(w2i))
    # t2i = defaultdict(lambda: len(t2i))
    PAD = w2i["<pad>"]  # 0
    UNK = w2i["<unk>"]  # 1
    EOS = w2i["<eos>"]  # 2
    SOS = w2i["<sos>"]  # 3
    dialog_idx = 0
    _ = w2i['$u']
    _ = w2i['$s']
    current_profile = []

    global_memory = {}
    # g = []

    for line in bytez.splitlines():
        if len(line) == 0:
            context = []
            time = 1
            sentinel = []
            idx = []
            dialog_idx += 1
            current_profile = []


        elif '\t' not in line:
            if line.split(' ')[0] == '1':
                temp = [[word, '$u', 't' + str(time)] for word in line.split(' ')[1:]]
                current_profile = line.split(' ',1)[1]
                # print(current_profile)
                profile_memory.append(temp)
                for w in temp:
                    for z in w:
                        _ = w2i[z]

                if current_profile not in global_memory.keys():
                    global_memory[current_profile] = []

            else:
                nid, sent = line.split(' ', 1)
                sent_new = []
                sent_token = sent.split(' ')

                if sent_token[1] == "R_rating":
                    sent_token_2 = sent_token
                    sent_token_2.extend(["<pad>"] * (3 - len(sent_token)))
                else:
                    sent_token_2 = sent_token[::-1]
                    sent_token_2.extend(["<pad>"] * (3 - len(sent_token)))
                sent_new.append(sent_token_2)
                context.extend(sent_new)

                for s in sent_new:
                    for w in s:
                        _ = w2i[w]

        else:
            sentinel = []  # gate
            idx = []  # index

            user, bot = line.split('\t')
            user = user.split(' ')[1:]  # list of words in user utterance

            context.extend([[word, '$u', 't' + str(time)] for word in user])

            _ = w2i['t' + str(time)]
            for w in user:
                _ = w2i[w]

            for word in bot.split(' '):
                index = [i for i, w in enumerate(context) if w[0] == word]
                # print(index)
                idx.append(max(index) if index else len(context))
                sentinel.append(bool(index))

            context_new = context.copy()

            context_new.extend([['$$$$'] * 3])

            memory.append([context_new, bot, idx, sentinel, dialog_idx, profile_memory[-1]])  ##### final output
            # g.append(0)
            for k in global_memory.keys():
                if k == current_profile:
                    global_memory[current_profile].extend(context_new)

            # global_memory[current_profile].append([context_new, bot, idx, sentinel, dialog_idx, profile_memory[-1]])

            context.extend([[word, '$s', 't' + str(time)] for word in bot.split(' ')])

            for w in bot.split(' '):
                _ = w2i[w]

            _ = w2i['t' + str(time)]
            time += 1
    # pdb.set_trace()
    # maxl = len(memory)
    # for k in global_memory.keys():
    #     templ = len(global_memory[k])
    #     if templ != maxl:
    #         global_memory[k] = [0]*(maxl - templ) + global_memory[k]

    gcontext_seq = global_memory

    gnew_context_seq = {}

    for k in gcontext_seq.keys():
        gnew_context_seq[k] = []
        for cg in gcontext_seq[k]:
            lg = []
            for word in cg:
                lg.append(w2i[word])
            gnew_context_seq[k].append(lg)

    for k in gnew_context_seq.keys():
        gnew_context_seq[k] = torch.from_numpy(np.array(gnew_context_seq[k]))




    return gnew_context_seq, w2i



