# -*- coding: utf-8 -*-
import os
import sys
import torch
import pickle

from transformer_lm.lib.util import Config, load_data
from transformer_lm.lib.data_preprocess import Vocab, preprocessor
from lib.bart_pretrain import BartPretrainingLM

file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)

# load configs
dconf_path = 'config/data.json'
mconf_path = 'config/lm.json'
dconf = Config(dconf_path)
mconf = Config(mconf_path)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print('Using device:', device)

try:
    with open('preprocessed_data.pickle', 'rb') as f:
        saved_obj = pickle.load(f)
        corpus, vocab = saved_obj
except:
    # load & preprocess corpus
    corpus = preprocessor(load_data(dconf.train_path), lang='ko')

    # load vocab
    vocab = Vocab(dconf.min_cnt)
    vocab.load(corpus)

    # save data as pickle
    with open('preprocessed_data.pickle', 'wb') as f:
        pickle.dump([corpus, vocab], f, protocol=pickle.HIGHEST_PROTOCOL)

assert all([corpus, vocab])


# load translator and train
bart = BartPretrainingLM(vocab, dconf, mconf, device)
bart.train(corpus)

# save model
bart.save('trained_keep.pth')
mconf.save(mconf_path)
