## 高频词（TF）特征模式

高频词特征模式主要关注词在文档中的出现频率，通常用词频（TF）来表示。公式如下：
$$
\text{TF}(t, d) = \frac{\text{词 } t \text{ 在文档 } d \text{ 中出现的次数}}{\text{文档 } d \text{ 中的总词数}}
$$
- t 表示词（term）。
- d 表示文档（document）。

## TF-IDF特征模式

TF-IDF特征模式综合考虑了**词频（TF）**和**逆文档频率（IDF）**，公式如下：
$$
\text{TF-IDF}(t, d) = \text{TF}(t, d) \times \text{IDF}(t)
$$

其中：
$$
\text{TF}(t, d) = \frac{\text{词 } t \text{ 在文档 } d \text{ 中出现的次数}}{\text{文档 } d \text{ 中的总词数}}
$$
$$
\text{IDF}(t) = \log \frac{\text{语料库文档总数}}{\text{包含词 } t \text{ 的文档数} + 1}
$$

- $\text{IDF}(t) $ 中的分母加1是为了避免分母为0的情况。
- $\log $ 是对数函数，通常使用自然对数（ln）或以10为底的对数。

## 特征模式切换方法

根据具体需求，可以选择使用**高频词（TF）特征模式**或**TF-IDF特征模式**：

- 如果希望更关注词在文档中的出现频率，可以使用**高频词（TF）特征模式**。
- 如果希望综合考虑词的全局重要性（即词在语料库中的分布情况），则使用**TF-IDF特征模式**。

在实际应用中，可以通过以下方式实现两种特征模式的切换：

```python
import math

def calculate_tf(word_count, total_words):
    """
    计算词频（TF）
    :param word_count: 词在文档中的出现次数
    :param total_words: 文档中的总词数
    :return: 词频（TF）
    """
    return word_count / total_words

def calculate_idf(total_documents, documents_with_word):
    """
    计算逆文档频率（IDF）
    :param total_documents: 语料库文档总数
    :param documents_with_word: 包含该词的文档数
    :return: 逆文档频率（IDF）
    """
    return math.log((total_documents + 1) / (documents_with_word + 1))

def calculate_tfidf(tf, idf):
    """
    计算TF-IDF
    :param tf: 词频（TF）
    :param idf: 逆文档频率（IDF）
    :return: TF-IDF值
    """
    return tf * idf

# 示例：根据需求选择特征模式
use_tfidf = True  # 设置为True时使用TF-IDF，设置为False时使用TF

if use_tfidf:
    tfidf_value = calculate_tfidf(tf_value, idf_value)
    print(f"TF-IDF值为: {tfidf_value}")
else:
    print(f"TF值为: {tf_value}")