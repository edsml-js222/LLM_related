# LLM_related
**The repository is mainly used to record the path of my lerning of some basic LLM-related knowledge🚀**

## The Structure of each point being recorded

- What it is？(・∀・？)
- Why do we need it? 🤔
- How to code it? 💻
- Potential 八股文hhh😀

## Already DONE!😊
* Transformer structure and their optimized technology
    * Tokenizer
        * BPE (Byte-Pair Encoding) & BBPE (Byte-level Byte)
    * Normalization
        * Layer Norm (why not Batch Norm)
        * RMSNorm
        * Pre-norm & Post-norm
    * Positional Encoding -- [code](transformer/pe.py) & [concepts]()
        * Sinusoidal
        * ALiBi (Attention with Linear Bias)
        * RoPE (Rotary Positional Encoding)
    * Attention
        * Softmax & safe-softmax
        * Masked attention
        * MHA (Multi-Head Attention)
        * MQA (Multi-Query Attention)
        * GQA (Group Query Attention)
        * KV-cache
        * Flash attention
        * Paged attention (vLLM)
    * FFN (Feed Forward Network)
        * SwiGLU

## TO BE Done...💪
* Agent
* RAG
* Deepspeed