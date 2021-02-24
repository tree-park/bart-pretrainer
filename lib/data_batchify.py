import torch
from torch.utils.data import Dataset


class Corpus(Dataset):

    def __init__(self, data_set):
        self._data = data_set

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx: int):
        return None


def collate_fn(batch):
    return None
