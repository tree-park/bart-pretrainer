import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformer_lm.lib.model.transformer import mask_not_pad
from lib.model.layer import BartEncoderLayer, BartDecoderLayer, BartEmbedding


class BartModel(nn.Module):

    def __init__(self, vocab_size, d_m, attn_heads, n):
        super(BartModel, self).__init__()

        # enc_layer
        self.encoder = BartEncoder(vocab_size, d_m, attn_heads, n)
        # dec_layer
        self.decoder = BartDecoder(vocab_size, d_m, attn_heads, n)
        self.linear = nn.Linear(d_m, vocab_size)

    def forward(self, inp_batch, dec_inp_batch, inp_mask, dec_inp_mask):
        """

        Args:
            inp_batch (Tensor): [batch_size, maxlen]
            dec_inp_batch (Tensor): [batch_size, maxlen]
            inp_mask (Tensor): [batch_size, maxlen]
            dec_inp_mask (Tensor): [batch_size, maxlen]

        Returns:

        """
        src_mask = mask_not_pad(inp_batch)

        # [batch size, maxlen, d_m]
        enc_outputs = self.encoder(inp_batch, src_mask)
        dec_output = self.decoder(dec_inp_batch, dec_inp_mask, enc_outputs[0], inp_mask)

        # [batch size, maxlen, vocab_size]
        lm_enc = self.linear(dec_output)
        # [batch size, num_mask, vocab_size]
        rst_1 = [F.log_softmax(lm_enc[i][posi], dim=-1) for i, posi in enumerate(inp_mask)]

        return rst_1


class BartEncoder(nn.Module):
    def __init__(self, vocab_size, d_m, attn_heads, n):
        super(BartEncoder, self).__init__()
        # embedding
        self.inp_emb = BartEmbedding(vocab_size, d_m)

        self.enc_layers = nn.ModuleList(
            [BartEncoderLayer(d_m, d_m*4, attn_heads) for _ in range(n)])

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
            [BartDecoderLayer(d_m, d_m*4, attn_heads) for _ in range(n)])

    def forward(self, dec_inp_batch, dec_inp_mask, enc_hid_state, inp_mask):
        # [batch size, maxlen, d_m]
        i_emb = self.inp_emb(dec_inp_batch)
        dec = i_emb
        for layer in self.dec_layers:
            # [batch size, maxlen, d_m]
            dec = layer(dec, dec_inp_mask, enc_hid_state, inp_mask)
        return dec
