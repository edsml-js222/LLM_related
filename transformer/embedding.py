import torch.nn as nn
import math
import torch

class Embedding(nn.Module):
    def __init__(self, vocab_size, d_model):
        super(Embedding, self).__init__()
        self.lut = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model
    
    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model) # [batch_size, seq_len, d_model]

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, seq_len_max=5000, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        position_ids = torch.arange(0, seq_len_max).unsqueeze(-1) # [5000,1]
        div_part = torch.exp(- torch.arange(0, d_model, 2) * math.log(10000) / d_model)
        pe = torch.zeros(seq_len_max, d_model)
        pe[:, 0::2] = torch.sin(position_ids * div_part)
        pe[:, 1::2] = torch.cos(position_ids * div_part) # [seq_len_max, d_model]
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x): # [batch_size, seq_len, d_model]
        x = x + self.pe[:, :x.size()[1], :]
        return self.dropout(x)
    
class Embedding(nn.Module):
    def __init__(self, vocab_size, d_model):
        super(Embedding, self).__init__()
        self.all_vocab_emb = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model
    
    def forward(self, x):
        return self.all_vocab_emb(x) * math.sqrt(self.d_model) # [batch_size, seq_len, d_model]

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, seq_len_max=5000, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(d=dropout)
        position_ids = torch.arange(0, seq_len_max).unsqueeze(-1)
        div_term = torch.exp(-torch.arange(0, d_model, 2) * math.log(10000) / d_model)
        pe = torch.zeros(seq_len_max, d_model)
        pe[:, 0::2] = torch.sin(position_ids * div_term)
        pe[:, 1::2] = torch.cos(position_ids * div_term)
        pe = pe.unsqueeze(0) # add batch dimention
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:, :x.size()[1], :]
        return self.dropout(x)
    
class RotaryPositionalEncoding(nn.Module):
    def __init__(self, d_model, base=10000):
        super(RotaryPositionalEncoding, self).__init__()
        self.inv_freq = 1.0 / (base ** (torch.arange(0, d_model, 2, dtype=torch.int64).float() / d_model))
    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, -1) # [batch_size, d_model//2, 1]
        position_ids_expanded = position_ids[:, None, :].float() # [batch_size, 1, seq_len]
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != 'mps' else 'cpu'
        with torch.autocast(device_type=device_type, enable=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1,2) # [batch_size, seq_len, d_model//2]
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)
    
def rotate_half(x):
    """Rotate half the hidden dims of the input."""
    x1 = x[..., :x.shape[-1]//2]
    x2 = x[..., x.shape[-1]//2:]
    return torch.cat((-x2, x1), dim=-1)

def applu_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    """
    Apply rotary positional embedding to the query and key tensors.

    Args:
        q, k (torch.tensor): query and key tensors
        cos, sin (torch.tensor): cosine and sine part of the rotary embedding
        unsqueeze_dim (int, *optional*, defalut to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos and sin
            so that they can be properly broadcasted to the dimensions of q and k. For example, note that
            the cos and sin have the shape [batch_size, seq_len, hidden_dim]. Then, if q and k have the shape
            [batch_size, num_heads, seq_len, d_model], then setting unsqueeze_dim=1 makes cos and sin broadcastable
            to the shape of q and k. Similarly, if q and k have the shape [batch_size, seq_len, num_heads, d_k], then 
            setting unsqueeze_dim=2. 
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed