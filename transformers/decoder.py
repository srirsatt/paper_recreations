import torch
import torch.nn as nn
import torch.optim as optim
import urllib.request
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
val_data = data[:n] # validate

block_size = 64 # "context" size, for decoder indexing
batch_size = 32

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y


class FeedForward(nn.Module):
    def __init__(self):
        super().__init__() 
        self.lin1 = nn.Linear(512, 2048)
        self.lin2 = nn.Linear(2048, 512)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.lin2(self.relu(self.lin1(x)))
        return x

