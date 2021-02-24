import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformer_lm.lib.model.transformer import mask_not_pad, mask_get_dec
from lib.model.layer import BartEncoderLayer, BartDecoderLayer, BartEmbedding


class BartModel(nn.Module):

    def __init__(self, vocab_size, d_m, attn_heads, n, sep_idx, n_sent):
        super(BartModel, self).__init__()
        self.sep_idx = sep_idx
        # enc_layer
        self.encoder = BartEncoder(vocab_size, d_m, attn_heads, n)
        # dec_layer
        self.decoder = BartDecoder(vocab_size, d_m, attn_heads, n)
        self.linear = nn.Linear(d_m, vocab_size)
        self.linear2 = nn.Linear(d_m, n_sent)

    def forward(self, inp_batch, inp_mask):
        """

        Args:
            inp_batch (Tensor): [batch_size, maxlen]
            dec_inp_batch (Tensor): [batch_size, maxlen]
            inp_mask (Tensor): [batch_size, maxlen]
            dec_inp_mask (Tensor): [batch_size, maxlen]

        Returns:

        """
        src_mask = mask_not_pad(inp_batch)
        sep_tokens = (inp_batch == self.sep_idx).unsqueeze(1)
        dec_inp_batch = shift_to_right(inp_batch)
        trg_mask = mask_get_dec(dec_inp_batch)

        # [batch size, maxlen, d_m]
        enc_output = self.encoder(inp_batch, src_mask)
        dec_output = self.decoder(dec_inp_batch, enc_output, inp_mask, trg_mask)

        # [batch size, maxlen, vocab_size]
        lm_enc = self.linear(dec_output)
        # [batch size, num_mask, vocab_size]
        rst_1 = [F.log_softmax(lm_enc[i][posi], dim=-1) for i, posi in enumerate(inp_mask)]

        # Sentence Permutation
        # [batch size, d_m]
        cls = torch.stack([dec_output[i][posi] for i, posi in enumerate(sep_tokens)])
        # [batch size, 2]
        rst_2 = F.log_softmax(self.linear2(cls), dim=-1)

        return rst_1, rst_2


class BartEncoder(nn.Module):
    def __init__(self, vocab_size, d_m, attn_heads, n):
        super(BartEncoder, self).__init__()
        # embedding
        self.inp_emb = BartEmbedding(vocab_size, d_m)

        self.enc_layers = nn.ModuleList(
            [BartEncoderLayer(d_m, d_m * 4, attn_heads) for _ in range(n)])

    def forward(self, inp_batch, src_mask):
        # [batch size, maxlen, d_m]
        i_emb = self.inp_emb(inp_batch)
        # Encoder
        enc = i_emb
        for layer in self.enc_layers:
            # [batch size, maxlen, d_m]
            enc = layer(enc, src_mask)
        return enc


class BartDecoder(nn.Module):
    def __init__(self, vocab_size, d_m, attn_heads, n):
        super(BartDecoder, self).__init__()
        # embedding
        self.dec_inp_emb = BartEmbedding(vocab_size, d_m)
        self.dec_layers = nn.ModuleList(
            [BartDecoderLayer(d_m, d_m * 4, attn_heads) for _ in range(n)])

    def forward(self, dec_inp_batch, enc_hid_state, inp_mask, trg_mask):
        # [batch size, maxlen, d_m]
        i_emb = self.inp_emb(dec_inp_batch)
        dec = i_emb
        for layer in self.dec_layers:
            # [batch size, maxlen, d_m]
            dec = layer(dec, trg_mask, enc_hid_state, inp_mask)
        return dec


def shift_to_right(inp):
    """

    Args:
        inp ():

    Returns:

    """
    return None