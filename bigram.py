import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 32
block_size = 8
max_iters = 3000
eval_interval = 300
learning_rate = 1e-2
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
eval_iters = 200


# load data
with open("data/tiny_shakespeare.txt", "r") as f:
    text = f.read()

# tokenizer
chars = sorted(list(set(text)))
vocab_size = len(chars)

c2i = {c: i for i, c in enumerate(chars)}
i2c = {i: c for i, c in enumerate(chars)}

encode = lambda s: [c2i[c] for c in s]
decode = lambda l: "".join([i2c[i] for i in l])

# train and test data
data = torch.tensor(encode(text), dtype=torch.long)
n = int(len(data) * 0.9)

train_data = data[:n]
val_data = data[n:]

# data loader
def get_batch(split):
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for i in range(eval_iters):
            x, y = get_batch(split)
            logitis, loss = model(x, y)
            losses[i] = loss.item()
        out[split] = losses.mean()
    model.train()

# model
class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        
        self.token_embedding_table =  nn.Embedding(vocab_size, vocab_size)


    def forward(self, idx, targets=None):
        # idx and targets are both (B, T) tensor of integers
        logits = self.token_embedding_table(idx) # (B, T, C)
    
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss
    

    def generate(self, idx, max_new_tokens):
        # idx is a (B, T) tensor of indives in the current context
        for _ in range(max_new_tokens):
                
            # get predictions for the next token
            logits, _ = self(idx)

            # only focus on the last one
            logits = logits[:, -1, :]
            
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)

            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            
            # append the new token to the context
            idx = torch.cat([idx, idx_next], dim=-1)
        
        return idx


model = BigramLanguageModel(vocab_size)
m = model.to(device)

print(f"Number of parameters: {sum(p.numel() for p in m.parameters())}")

optimizer = torch.optim.AdamW(m.parameters(), lr=0.001)

for iter in range(max_iters):

    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step: {iter:04d} train loss: {losses['train']:.4f} val loss: {losses['val']:.4f}")

    xb, yb = get_batch("train")

    logitis, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate text
context = torch.zeros(1, 1, dtype=torch.long).to(device)
ret = m.generate(context, max_new_tokens=500)[0].tolist()
print(decode(ret))
