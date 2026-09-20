import torch
import torch.nn as nn
import torch.optim as optim
import urllib.request
from torch.utils.data import DataLoader, Dataset


urllib.request.urlretrieve(
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
    "input.txt" #save name
)

