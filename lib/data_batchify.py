import torch
from torch.utils.data import Dataset


class Corpus(Dataset):

    def __init__(self, data_set):
        self._data = data_set

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx: int):
        inp, inp_mask, slm_trg, sent_perm_trg = self._data[idx][0], self._data[idx][1], \
                                                self._data[idx][2], self._data[idx][3]

        return inp, inp_mask, slm_trg, sent_perm_trg


def collate_fn(batch):
    inp, inp_mask, slm_trg, sent_perm_trg = zip(*batch)
    # inp_batch, dec_inp_batch, inp_mask, dec_inp_mask = zip(*batch)
    pad_inp = torch.nn.utils.rnn.pad_sequence(inp, batch_first=True)
    return pad_inp, inp_mask, slm_trg, sent_perm_trg
