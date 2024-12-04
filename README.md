# LLM_related
**The repository is mainly used to record the path of my lerning of some basic LLM-related knowledge🚀**

## The Structure of each point being recorded
- What it is？(・∀・？)👀
- Why do we need it? 🤔
- How to code it? 💻
- Potential 八股文hhh😀

## In Progress...⛏️！
* Transformer structure and their optimized technology
    * Tokenizer -- [concepts](transformer/docs/tokenizer.md)
        * BPE (Byte-Pair Encoding) & BBPE (Byte-level Byte)
    * Normalization -- [code](transformer/codes/normalization.py) & [concepts](transformer/docs/normalization.md)
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

## UpdateLog
[03/12/2024 Mon] Tokenization (concepts)-- ✅ <br>
[04/12/2024 Tues] Normalization (code & concepts) -- ✅