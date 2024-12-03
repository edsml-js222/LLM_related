import torch.nn as nn
import math
import torch
    
class Embedding(nn.Module):
    def __init__(self, vocab_size, d_model):
        super(Embedding, self).__init__()
        self.all_vocab_emb = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model
    
    def forward(self, x):
        return self.all_vocab_emb(x) * math.sqrt(self.d_model) # [batch_size, seq_len, d_model]
