# Tokenizer（分词器）
* **What is Tokenizer**: 分词器是帮助我们将一句自然语言分成一个一个词的工具

* **Why need Tokenizer**: 计算机无法直接处理自然语言，需要首先将自然语言转换为计算机可
以理解的向量形式，而如果我们直接把一整个句子进行向量化，那么会丢失很多词语之间的语义信息。又因为词
是自然语言中的最小单位，所以我们需要先用分词器将句子拆分成一个个的词，然后将词进行向量化，最后再将
向量化后的词拼接成完整的句子。<br>
举个例子，我们有一句话"I am iron man", 分词后为["I", "am", "iron", "man"]，然后我们将每个词进行向量化为768维度的向量，那么最后原句经过分词和向量化后会变成形状为[4, 768]的矩阵，这个矩阵就是计算机可以理解的形式了。

* **NEXT**: 了解到这里我会疑惑，ok，我了解了我们需要分词，但是根据什么规则去分词呢？为什么"I am iron man", 分词后就是["I", "am", "iron", "man"]？首先分词规则需要确定的是按照什么<span style='color:red;'>**颗粒度**</span>去对句子做切分，通常有三种:Word(单词), Subword(字词), Char(字符)。而上面这种就对应了Word颗粒度进行切分，这种切分方法的弊端是如果语料中有很多低频词的话，词表[1](#reference-1)

## 1. Byte Pair Encoding (BPE)
* **What is BPE**
    * 
* **Why need BPE**
## 2. Byte-level BPE (BBPE)

## 3. WordPiece

## 4. Unigram


## Reference
### 1. https://zhuanlan.zhihu.com/p/649030161