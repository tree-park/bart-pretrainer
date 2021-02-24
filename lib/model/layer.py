import torch
import torch.nn as nn

from transformer_lm.lib.model.layers.sublayer import MultiHeadAttention, PositionWiseFFLayer
from transformer_lm.lib.model.layers.embedding import WordEmbedding, positional_embedding


class BartEmbedding(nn.Module):
    def __init__(self, vocab_size, d_m):
        super(BartEmbedding, self).__init__()
        self.word_emb = WordEmbedding(vocab_size, d_m)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, inp):
        """
        inp [bsize, maxlen]
        """
        # [bsize, maxlen, emb_dim]
        idx_emb = self.word_emb(inp)
        pe_emb = positional_embedding(idx_emb.size(0), idx_emb.size(1), idx_emb.size(2))
        emb = idx_emb + pe_emb
        return self.dropout(emb)


class BartEncoderLayer(nn.Module):
    def __init__(self, d_m, d_ff, n_head):
        super(BartEncoderLayer, self).__init__()
        self.multi_attn = MultiHeadAttention(d_m, n_head)
        self.pw_ff = PositionWiseFFLayer(d_m, d_ff)

    def forward(self, enc_emb, src_mask):
        """
        Args:
            enc_emb (Tensor): [batch size, maxlen, d_m]
            src_mask (Tensor): [batch size, 1, maxlen]
        Returns: [batch size, maxlen, d_m]
        """
        # Sub-layer 1
        out = self.multi_attn(enc_emb, enc_emb, enc_emb, src_mask)
        # Sub-layer 2
        out = self.pw_ff(out)
        return out


class BartDecoderLayer(nn.Module):
    def __init__(self, d_m, d_ff, n_head):
        super(BartDecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_m, n_head)
        self.encoder_attn = MultiHeadAttention(d_m, n_head)
        self.pw_ff = PositionWiseFFLayer(d_m, d_ff)

    def forward(self, dec_emb, dec_inp_mask, enc_hid_state, inp_mask):
        """
        Args:
            dec_emb (Tensor): [batch size, maxlen, d_m]
            dec_inp_mask (Tensor): [batch size, maxlen, maxlen]
            enc_hid_state (Tensor): [batch size, maxlen, d_m]
            inp_mask (Tensor): [batch size, 1, maxlen]
        Returns: [batch size, maxlen, d_m]
        """
        # Sub-layer 1 - masked self attention
        out = self.self_attn(dec_emb, dec_emb, dec_emb, dec_inp_mask)
        # Sub-layer 2 - encoder-decoder attention
        out = self.encoder_attn(out, enc_hid_state, enc_hid_state, inp_mask)
        # Sub-layer 3
        out = self.pw_ff(out)
        return out
