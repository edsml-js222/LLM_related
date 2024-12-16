# Positional Encoding（位置编码）
* Why need positional encoding?<br>
Transformer架构中用到的自注意力机制本身不具备处理序列数据的顺序信息的能力，但这个在语言问题中是很重要的，因为一句话中的不同词的顺序改变之后往往句子的意义会天差地别。所以就需要位置编码来给数据添加顺序信息，以便模型获取理解。<br>
<br>
下文围绕两类常见的位置编码类型：绝对位置编码和相对位置编码展开

## 绝对位置编码
* 定义：为序列中的每个数据的每个维度都分配一个唯一的编码，表示该位置的绝对位置
* 例子：
    * sinusoidal(通过正余弦函数实现)
        * 公式：
        $$
        PE(pos, 2i) = \sin\left(\frac{pos}{10000^{\frac{2i}{d_{model}}}}\right)
        $$
        $$
        PE(pos, 2i + 1) = \cos\left(\frac{pos}{10000^{\frac{2i}{d_{model}}}}\right)
        $$

            pos: 词向量位于序列中的位置
            i: 单个词向量的每个维度<br>
            d_model: 隐藏空间的维度大小


## 混合位置编码
* 定义：通常指代RoPE(旋转位置编码)，RoPE属于一种融合范式的位置编码他将绝对位置编码和相对位置编码结合到了一起，他在给向量编码位置信息的时候也是用的绝对位置编码，但是attention模块在计算向量内积的时候也等价于相对位置编码。编码的时候用的是
$x' = x * e^{i * pos * \theta}$ (显示的依赖绝对位置pos)，但是内积计算的时候，计算结果只和(pos1 - pos2)也就是向量之间的相对位置有关系。（这里的计算的时候用了复数计算来推导，两个二维向量的内积，当把他们当成复数看的时候，可以看成一个复数和另一个复数的共轭的乘积的实部）
* 例子：
    * RoPE (Rotary positional encoding):公式参考如下，但是具体的
    $$
    PE(pos) = \begin{pmatrix} \cos(pos * \theta) & -\sin(pos * \theta) \\ \sin(pos * \theta) & \cos(pos * \theta) \end{pmatrix} \cdot \text{Embedding}
    $$
    $$\theta = 10000^{-\frac{2i}{d}}$$