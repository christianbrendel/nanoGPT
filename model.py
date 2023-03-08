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
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class AttentionHead(nn.Module):

    def __init__(self, d_model=D_MODEL, d_head=D_HEAD, dropout=DROPOUT):
        super().__init__()
        self.key = nn.Linear(d_model, d_head, bias=False)
        self.query = nn.Linear(d_model, d_head, bias=False)
        self.value = nn.Linear(d_model, d_head, bias=False)
        self.dropout = nn.Dropout(dropout)

        self.register_buffer("tril", torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE)))


    def forward(self, x):
        
        B, T, C = x.shape

        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        wei = q @ k.transpose(-2, -1)  * (C ** -0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))        
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        out = wei @ v
        return out


class MultiHeadAttention(nn.Module):

    def __init__(self, d_model=D_MODEL, n_heads=N_HEADS, d_head=D_HEAD, dropout=DROPOUT):
        super().__init__()
        self.heads = nn.ModuleList([AttentionHead(d_model=d_model, d_head=d_head) for _ in range(n_heads)])
        self.proj = nn.Linear(n_heads * d_head, d_model) # Projection layer going back into residual pathway
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
            nn.Linear(d_inner, d_model), # Projection layer going back into residual pathway
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)
    

class Block(nn.Module):

    def __init__(self, d_model=D_MODEL, n_heads=N_HEADS, d_head=D_HEAD, d_inner=D_INNER):
        super().__init__()
        self.mha = MultiHeadAttention(d_model=d_model, n_heads=n_heads, d_head=d_head)
        self.ln1 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model=d_model, d_inner=d_inner)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x):
        x = x + self.mha(self.ln1(x))
        x = x + self.ff(self.ln2(x))  
        return x


class LanguageModel(nn.Module):

    def __init__(self, vocab_size=VOCAB_SIZE, d_model=D_MODEL, n_heads=N_HEADS, n_blocks=N_BLOCKS):
        super().__init__()
        
        self.token_embedding_table =  nn.Embedding(vocab_size, d_model)
        self.position_embedding_table = nn.Embedding(BLOCK_SIZE, d_model)
        
        self.blocks = nn.Sequential(*[
            Block(d_model, n_heads=n_heads) for _ in range(n_blocks)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)


    def forward(self, idx, targets=None):

        B, T = idx.shape
        # idx and targets are both (B, T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B, T, C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T, C)
        x = tok_emb + pos_emb # (B, T, C)
        x = self.blocks(x) # (B, T, C)
        x = self.ln_f(x) # (B, T, C)
        logits = self.lm_head(x) # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss