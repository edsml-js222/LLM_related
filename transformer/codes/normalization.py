import torch.nn as nn
import torch
class RMSNorm(nn.Module):
    """The same as Llama RMSNorm"""
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.varience_epsilon = eps
    
    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        varience = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(varience + self.varience_epsilon)
        return self.weight * hidden_states.to(input_dtype)

if __name__ == "__main__":
    batch_size = 2
    seq_len = 4
    hidden_size = 16
    input_tensor = torch.rand(batch_size,seq_len,hidden_size)
    print(f"input_tensor: {input_tensor}")
    rms = RMSNorm(hidden_size)
    print(f"weight: {rms.weight}")
    print(f"variance: {input_tensor.pow(2).mean(-1, keepdim=True)}")
    after_rms = rms(input_tensor)
    print(f"after_rms: {after_rms}")
