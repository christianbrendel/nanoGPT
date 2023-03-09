import torch
import torch.nn as nn
import torch.nn.functional as F

D_MODEL = 384
N_HEADS = 6
D_HEAD = D_MODEL // N_HEADS
D_INNER = 4 * D_MODEL
N_BLOCKS = 6
DROPOUT = 0.1

VOCAB_SIZE = 26
BLOCK_SIZE = 256


class AttentionHead(nn.Module):
    def __init__(
        self, d_model=D_MODEL, d_head=D_HEAD, block_size=BLOCK_SIZE, dropout=DROPOUT
    ):
        super().__init__()
        self.key = nn.Linear(d_model, d_head, bias=False)
        self.query = nn.Linear(d_model, d_head, bias=False)
        self.value = nn.Linear(d_model, d_head, bias=False)
        self.dropout = nn.Dropout(dropout)

        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape

        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        wei = q @ k.transpose(-2, -1) * (C**-0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        out = wei @ v
        return out


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        d_head=D_HEAD,
        block_size=BLOCK_SIZE,
        dropout=DROPOUT,
    ):
        super().__init__()
        kwargs = dict(
            d_model=d_model, d_head=d_head, block_size=block_size, dropout=dropout
        )
        self.heads = nn.ModuleList([AttentionHead(**kwargs) for _ in range(n_heads)])
        self.proj = nn.Linear(n_heads * d_head, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class FeedForward(nn.Module):
    def __init__(self, d_model=D_MODEL, d_inner=D_INNER, dropout=DROPOUT):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_inner),
            nn.ReLU(),
            nn.Linear(d_inner, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(
        self,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        d_head=D_HEAD,
        d_inner=D_INNER,
        block_size=BLOCK_SIZE,
        dropout=DROPOUT,
    ):
        super().__init__()
        self.mha = MultiHeadAttention(
            d_model=d_model,
            n_heads=n_heads,
            d_head=d_head,
            block_size=block_size,
            dropout=dropout,
        )
        self.ln1 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model=d_model, d_inner=d_inner, dropout=dropout)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x):
        x = x + self.mha(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class LanguageModel(nn.Module):
    def __init__(
        self,
        vocab_size=VOCAB_SIZE,
        block_size=BLOCK_SIZE,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        d_head=D_HEAD,
        n_blocks=N_BLOCKS,
        d_inner=D_INNER,
        dropout=DROPOUT,
    ):
        super().__init__()

        # embedding tables
        self.token_embedding_table = nn.Embedding(vocab_size, d_model)
        self.position_embedding_table = nn.Embedding(block_size, d_model)

        # transformer blocks
        block_args = dict(
            d_model=d_model,
            n_heads=n_heads,
            d_head=d_head,
            d_inner=d_inner,
            block_size=block_size,
            dropout=dropout,
        )
        self.blocks = nn.Sequential(*[Block(**block_args) for _ in range(n_blocks)])

        # output head
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)

        # register buffer for position embeddings
        self.register_buffer("timesteps", torch.arange(block_size))

    def forward(self, idx, targets=None):
        batch_size, sequence_length = idx.shape

        # token embeddings
        tok_emb = self.token_embedding_table(idx)

        # position embeddings
        pos_emb = self.position_embedding_table(self.timesteps[:sequence_length])

        # sum embeddings
        x = tok_emb + pos_emb

        # transformer blocks
        x = self.blocks(x)

        # final layer normalization
        x = self.ln_f(x)

        # language model head
        logits = self.lm_head(x)

        # loss
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss
