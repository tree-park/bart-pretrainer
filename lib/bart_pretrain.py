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

from transformer_lm.lib.util import load_data
from transformer_lm.lib.model.utils import LabelSmoothingLoss
from transformer_lm.lib.language_model import BasicLM
from transformer_lm.lib.data_preprocess import preprocessor, Vocab, TokenMarks
from lib.data_batchify import Corpus, collate_fn
from lib.model.bart import BartModel


class BartPretrainingLM(BasicLM):

    def __init__(self):
        super(BartPretrainingLM, self).__init__()

    def train(self):
        """
        - clean and load data
        - load data preprocessor
        - define model and loss
        - train by epoch
        Args:

        """

    def masking_words(self):
        """
        Masking text spans as a mask token
        15 % of words in text will be masked
        (refer from SpanBERT's Span Boundary Objective)
        """
        pass

    def dataset_form(self, corpus):
        pass
