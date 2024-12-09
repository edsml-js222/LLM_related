# Normalization (归一化)
* **What is Normalization**: 
    * 归一化指的是在指定的维度上将输入数据调整为标准正态分布的形式，比如layernorm和rmsnorm就是在训练样本的特征维度下进行归一化，而batchnorm是在批次维度下进行的归一化
* **Why need Normalization**
    * 主要是为了保证训练过程中数据的尺度范围能够保持一致，能够稳定和加速训练过程，也有助于改善收敛性，防止梯度消失和梯度爆炸等问题

## 1. Batch Norm
* **What is Batch Norm**: 
    * 公式:
    $$
    \hat{x_i} = \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}, \quad y_i = \gamma \hat{x_i} + \beta
    $$
    * 在批次维度下对数据进行归一化

## 2. Layer Norm
* **What is Layer Norm**
    * 公式:
    $$
    \hat{x_i} = \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}}, \quad y_i = \gamma \hat{x_i} + \beta
    $$    
    * 在**单个样本**的特征维度下进行归一化
* **Why need Layer Norm**
    * nlp任务需要处理可变序列长度的问题，batch norm是假设在batch维度下的数据都能参与稳定均值和方差的计算，但是nlp任务中输入序列的长度是变化的，往往通过padding的方式来让同一batch下的序列等长，padding往往会带来一些不连续性<br>
    ps：特别需要说明的是之前错误理解为layernorm是将句子中所有词的嵌入向量一起做归一化，但其实参考llama的rmsnorm实现发现，其实是在句子中的每单个词的嵌入向量维度下进行归一化。

## 3. RMS Norm
* **What is RMS Norm**
    * 公式：
    $$
    \text{RMSNorm}(x) = \frac{x_i}{\text{RMS}(x)} \times \gamma, \quad \text{RMS}(x) = \sqrt{\frac{1}{N} \sum_{i=1}^{N} x_i^2 + \epsilon}
    $$
    * RMS Norm和layer norm类似，不同的点在于移除了均值项，而改用均方根的方式来做归一化
* **Why need RMS Norm**
    * 主要rms norm的计算相比于layer norm要更加简单，省去了均值和方差的复杂计算，但是效果却和layer norm基本相近甚至有些情况下效果会更好。所以rms norm是一个有很高计算性价比的选择。

## 4. Pre-Norm vs Post-Norm
 * **Pre-Norm**: 在每个子层（如自注意力层或前馈层）之前进行归一化
    * 公式：
    $$
    \text{Output} = \text{Layer}( \text{LayerNorm}( \text{Input} ) ) + \text{Input}
    $$
* **Post-Norm**: 在每个子层之后进行归一化
    * 公式：
    $$
    \text{Output} = \text{LayerNorm}( \text{Layer}( \text{Input} ) + \text{Input} )
    $$
* **Why Pre-Norm**：主流的大模型训练一般都选择pre-norm的原因主要是pre-norm有更稳定的训练和更快的收敛效果
    * **训练稳定性**：pre-norm可以在训练过程中提供更好的稳定性，尤其是在深层网络中。他可以防止子层输出激活值的分布发生剧烈变化，减少梯度消失和梯度爆炸的风险。
    * **加快收敛**：由于子层的输入在进入子层之前就经过了归一化，模型的训练通常会更快收敛。