import torch
import torch.nn.functional as F
import torch.nn as nn
import math, copy

def clone(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])
def attention(q, k, v, dropout=None, mask=None):
    """COMPUTE CROSS DOT PRODUCTION"""
    d_k = q.size()[-1]
    atten_score = torch.matmul(q, k.transpose(-2,-1)) / math.sqrt(d_k)
    if not mask:
        atten_score = atten_score.masked_fill(mask==0, -1e9)
    p_atten = F.softmax(atten_score, dim=-1)
    if not dropout:
        p_atten = dropout(p_atten)
    atten_res = torch.matmul(p_atten, v)
    return atten_res, p_atten

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, d_model, dropout=0.1, mask=None):
        super(MultiHeadAttention).__init__()
        assert d_model % num_heads

        self.d_k = d_model // num_heads
        self.num_heads  = num_heads
        self.linears = clone(nn.Linear(d_model, d_model), 4)
        self.atten = None
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        if not mask:
            mask = mask.unsqueeze(1)
        nbatches = q.size()[0]

        q, k, v = [l(x).view(nbatches, -1, self.num_heads, self.d_k).transpose(1,2) for l, x in zip(self.linears, (q, k, v))]

        atten_res, self.atten = attention(q, k, v, dropout=self.dropout, mask=mask)

        atten_res = atten_res.transpose(1,2).contiguous().view(nbatches, -1, self.num_heads * self.d_k)
        return self.linears[-1](atten_res)







