import math

import torch
from torch import nn
import torch.nn.functional as F


def _make_linear(linear_factory, in_features, out_features, bias=True):
    # tiny helper so we can swap in the sharded linear for stage 3
    return linear_factory(in_features, out_features, bias=bias)


class MultiheadSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, linear_factory, dropout=0.0, seq_len=128):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.qkv = _make_linear(linear_factory, d_model, 3 * d_model, bias=True)
        self.proj = _make_linear(linear_factory, d_model, d_model, bias=True)
        self.dropout = dropout

        # causal mask so we don't peek at future tokens
        mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)
        self.register_buffer("causal_mask", mask, persistent=False)

    def forward(self, x):
        bsz, seq_len, _ = x.shape

        # project once, then split into q/k/v
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        # shape into heads
        q = q.view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_scores = attn_scores.masked_fill(
            self.causal_mask[:seq_len, :seq_len], float("-inf")
        )
        attn = F.softmax(attn_scores, dim=-1)
        if self.dropout > 0:
            attn = F.dropout(attn, p=self.dropout, training=self.training)

        # attention output back to (bsz, seq, d_model)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(bsz, seq_len, self.d_model)
        return self.proj(out)


class FeedForward(nn.Module):
    def __init__(self, d_model, dim_ff, linear_factory, dropout=0.0):
        super().__init__()
        self.fc1 = _make_linear(linear_factory, d_model, dim_ff, bias=True)
        self.fc2 = _make_linear(linear_factory, dim_ff, d_model, bias=True)
        self.dropout = dropout

    def forward(self, x):
        x = self.fc1(x)
        x = F.gelu(x)
        if self.dropout > 0:
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.fc2(x)


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, dim_ff, linear_factory, dropout=0.0, seq_len=128):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.attn = MultiheadSelfAttention(
            d_model,
            n_heads,
            linear_factory,
            dropout=dropout,
            seq_len=seq_len,
        )
        self.ff = FeedForward(d_model, dim_ff, linear_factory, dropout=dropout)
        self.dropout = dropout

    def forward(self, x):
        # standard pre-norm + residual
        x = x + self.attn(self.ln1(x))
        if self.dropout > 0:
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = x + self.ff(self.ln2(x))
        return x


class TransformerModel(nn.Module):
    def __init__(
        self,
        vocab_size=8192,
        d_model=512,
        n_layers=6,
        n_heads=8,
        dim_ff=2048,
        seq_len=128,
        dropout=0.0,
        linear_factory=None,
    ):
        super().__init__()
        if linear_factory is None:
            linear_factory = lambda in_f, out_f, bias=True: nn.Linear(
                in_f, out_f, bias=bias
            )

        # simple decoder-only transformer
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.seq_len = seq_len

        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, seq_len, d_model))

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_model,
                    n_heads,
                    dim_ff,
                    linear_factory,
                    dropout=dropout,
                    seq_len=seq_len,
                )
                for _ in range(n_layers)
            ]
        )

        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = _make_linear(linear_factory, d_model, vocab_size, bias=True)

        nn.init.normal_(self.pos_emb, mean=0.0, std=0.02)

    def forward(self, idx):
        bsz, seq_len = idx.shape
        if seq_len > self.seq_len:
            raise ValueError("seq_len is larger than configured model seq_len")

        # token + position embeddings
        x = self.tok_emb(idx) + self.pos_emb[:, :seq_len, :]
        # run through blocks
        for block in self.blocks:
            x = block(x)
        # final norm + vocab head
        x = self.ln_f(x)
        return self.lm_head(x)
