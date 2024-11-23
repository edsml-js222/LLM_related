import os
import sys
dir_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(dir_path)

from embedding import Embedding, PositionalEncoding, RotaryPositionalEncoding, rotate_half, apply_rotary_pos_emb
import torch

vocab_size = 10
d_model = 4
batch_size = 2
seq_len = 3

test_input = torch.randint(0, vocab_size, (batch_size, seq_len), dtype=torch.int64)
print(f"The shape of test_input is {test_input.shape}\n")

# Embedding test
print("*"*10 + "Embedding test" + "*"*10)
word_emb = Embedding(vocab_size, d_model)
test_input_emb = word_emb(test_input)
print(f"The shape after embedding should be [{batch_size}, {seq_len}, {d_model}]\nActually it is {test_input_emb.shape}")
print("*"*10 + 
      "Embedding test: " +
    (f"\033[92mPassed\033[0m" if test_input_emb.shape == torch.Size([batch_size, seq_len, d_model]) 
     else f"\033[91mFailed\033[0m") +
      "*" * 10)

# Sinusoidal Positional Encoding test
sinusoidal_pe = PositionalEncoding(d_model)
test_input_emb_with_sinusoidal_pe, with_dropout = sinusoidal_pe(test_input_emb)
print(f"The shape of sinusoidal positional encoding is: {sinusoidal_pe.pe.shape}\n")
print(f"The part of pe added to the test input is:\n{sinusoidal_pe.pe[:, :seq_len], :}\nshape:{sinusoidal_pe.pe[:, :seq_len, :].shape}")
print(f"The test input after embedding before adding pe is:\n{test_input_emb}\nshape:{test_input_emb.shape}")
print(f"The test input after adding pe is: \n{test_input_emb_with_sinusoidal_pe}\nshape:{test_input_emb_with_sinusoidal_pe.shape}")
print(f"The test input after dropout is: \n{with_dropout}\nshape:{with_dropout.shape}")
print(f"check for dropout output:\n{test_input_emb_with_sinusoidal_pe * 1 / (1 - 0.1)}")
#print(torch.rand_like(test_input_emb))
