import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

import math
import os
import sys
import random
import numpy as np

from transformer_lm.lib.language_model import BasicLM
from lib.data_batchify import Corpus, collate_fn
from lib.model.bart import BartModel


class BartPretrainingLM(BasicLM):

    def __init__(self, vocab, dconf, mconf, device):
        super(BartPretrainingLM, self).__init__(dconf, mconf)
        self.dconf = dconf
        self.mconf = mconf

        self.ko_vocab = vocab
        self.dataset = None
        self.dataload = None
        self.device = device

        self.model = BartModel(dconf.vocab_size,
                               mconf.d_m,
                               mconf.n_attn_heads,
                               mconf.n_layer,
                               dconf.sep_idx,
                               dconf.n_sent)
        self.loss = nn.CrossEntropyLoss()
        self.optim = optim.Adam(params=self.model.parameters(), lr=self.mconf.lr)
        self.lrscheder = optim.lr_scheduler.ReduceLROnPlateau(self.optim, patience=5)
        self.model.to(device)

    def train(self, corpus):
        """
        - clean and load data
        - load data preprocessor
        - define model and loss
        - train by epoch
        Args:

        """
        train_set = self.dataset_form(corpus)
        self.dataset = Corpus(train_set)
        self._dataload = DataLoader(self.dataset,
                                    batch_size=self.mconf.batch_size,
                                    num_workers=4, collate_fn=collate_fn)
        self.mconf.ko_size = len(self.ko_vocab)

        total_loss = 0
        total_acc = 0
        self.model.train()
        for epoch in tqdm(range(self.mconf.epoch), desc='epoch'):
            for i, batch in tqdm(enumerate(self._dataload), desc="step", total=len(self._dataload)):
                self.optim.zero_grad()
                # src, trg for Masked LM
                inp, inp_mask, slm_trg, sent_perm_trg = map(lambda ds: ds.to(self.device), batch)
                pred_slm, pred_sent_perm = self.model(inp, inp_mask)

                b_loss_sp = self.loss(pred_sent_perm, sent_perm_trg)
                b_loss_slm = sum([self.loss(pred, trg) for pred, trg in zip(pred_slm, slm_trg)])
                b_loss = b_loss_slm + b_loss_sp
                b_loss.backward()

                self.optim.step()

                # total_acc += accuracy(pred_lm, lm_trg)
                total_loss += b_loss.item()

            itersize = math.ceil(len(self.dataset) / self.mconf.batch_size)
            ppl = math.exp(total_loss / len(self.dataset))
            print(epoch, total_loss, total_acc / itersize, ppl)
            self.lrscheder.step(total_loss)
            total_loss = 0
        self.ko_vocab.to_idx2word()

    def masking_words(self):
        """
        Masking text spans as a mask token
        15 % of words in text will be masked
        (refer from SpanBERT's Span Boundary Objective)
        """
        pass

    def dataset_form(self, corpus):
        """
        Convert corpus to trainable element
        span masking + sentence permutation
        Args:
            corpus (list):

        Returns:
            inp: [[<srt>, tid0, ..<eos>]..]
            inp_mask: [[<srt>, 0, 1, 1, 0..<eos>]..]
            slm_trg: [[tidx, tidx..]..]
            sent_perm_trg: [n, 3, 0..]
        """
        pass
