import torch
import torch.nn as nn
import torch.optim as optim
import urllib.request
import math
from torch.utils.data import DataLoader, Dataset


urllib.request.urlretrieve(
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
    "input.txt" #save name
)

with open('input.txt', 'r') as f:
    text = f.read() # stream

chars = sorted(list(set(text)))
vocab_size = len(chars)

char_to_idx = {c: i for i, c in enumerate(chars)}
idx_to_char = {i: c for i, c in enumerate(chars)}
data = torch.tensor([char_to_idx[c] for c in text], dtype=torch.long)

n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:] # validate

block_size = 64 # "context" size, for decoder indexing
batch_size = 32

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y



class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__() 
        self.lin1 = nn.Linear(d_model, d_ff)
        self.lin2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.lin2(self.relu(self.lin1(x)))
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, d_k):
        super().__init__()
        # Q, K, V -> parameters
        # d_model = input/output dimension
        # num_heads = split for "multi-head" attention
        # d_k = d_model // num_heads (64)

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.num_heads = num_heads
        self.d_k = d_k
        self.d_model = d_model
        #output projection
    def forward(self, x, mask=None):
        batch, seq_len, d_model = x.shape
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)
        # [batch, seq_len, d_model]
        # split into 8 heads for parallelism on attention
        Q = Q.view(batch, seq_len, self.num_heads, self.d_k) # then split
        Q = Q.transpose(1, 2) # reorder
        K = K.view(batch, seq_len, self.num_heads, self.d_k)
        K = K.transpose(1, 2)
        V = V.view(batch, seq_len, self.num_heads, self.d_k)
        V = V.transpose(1, 2) 

        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.d_k) # scores computed per token pair

        # masking on scores
        mask = torch.tril(torch.ones(seq_len, seq_len)).to(x.device)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        weights = torch.softmax(scores, dim=-1)
        output = weights @ V
        # concatenate the heads back together

        output = output.transpose(1, 2).contiguous().view(batch, seq_len, self.d_model)

        output = self.W_o(output)
        return output


class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_k, d_ff):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads, d_k)
        self.feed_forward = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
    def forward(self, x, mask=None):
        x = x + self.attention(x, mask)
        x = self.norm1(x)
        x = x + self.feed_forward(x)
        x = self.norm2(x)
        return x













