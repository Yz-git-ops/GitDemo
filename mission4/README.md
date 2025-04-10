## 高频词（TF）特征模式

高频词特征模式主要关注词在文档中的出现频率，通常用词频（TF）来表示。公式如下：
$\text{TF}(t, d) = \frac{\text{词 } t \text{ 在文档 } d \text{ 中出现的次数}}{\text{文档 } d \text{ 中的总词数}}$ 
- t 表示词（term）。
- d 表示文档（document）。

## TF-IDF特征模式

TF-IDF特征模式综合考虑了**词频（TF）**和**逆文档频率（IDF）**，公式如下：
$\text{TF-IDF}(t, d) = \text{TF}(t, d) \times \text{IDF}(t)$ 

其中：
$\text{TF}(t, d) = \frac{\text{词 } t \text{ 在文档 } d \text{ 中出现的次数}}{\text{文档 } d \text{ 中的总词数}}$ 
$\text{IDF}(t) = \log \frac{\text{语料库文档总数}}{\text{包含词 } t \text{ 的文档数} + 1}$ 

- $\text{IDF}(t)$ 中的分母加1是为了避免分母为0的情况。
- $\log$ 是对数函数，通常使用自然对数（ln）或以10为底的对数。

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
```


## 附加5
```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE

# 模拟数据集
# 假设我们有一个包含垃圾邮件和普通邮件的数据集
np.random.seed(42)
data = {
    'feature1': np.random.rand(151),
    'feature2': np.random.rand(151),
    'label': [1] * 127 + [0] * 24  # 1 表示垃圾邮件，0 表示普通邮件
}
df = pd.DataFrame(data)

# 分离特征和标签
X = df[['feature1', 'feature2']]
y = df['label']

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 使用SMOTE进行过采样
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# 训练模型
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_resampled, y_resampled)

# 进行预测
y_pred = model.predict(X_test)

# 输出分类评估报告
report = classification_report(y_test, y_pred, target_names=['Normal', 'Spam'])
print("分类评估报告：")
print(report)
```
# 运行结果
<img src="https://github.com/Yz-git-ops/GitDemo/blob/main/mission4/image/simple%20balancing%20processing.png" width=“200”>

## 附加六
```python
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.datasets import make_classification
from imblearn.over_sampling import SMOTE

X, y = make_classification(n_samples=151, n_features=10, n_classes=2, weights=[0.2, 0.8], random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

smote = SMOTE(sampling_strategy='auto', random_state=42)
X_res, y_res = smote.fit_resample(X_train, y_train)

model = RandomForestClassifier(random_state=42)
model.fit(X_res, y_res)

y_pred = model.predict(X_test)

print(classification_report(y_test, y_pred))
```
# 运行结果
<img src="https://github.com/Yz-git-ops/GitDemo/blob/main/mission4/image/add%20model%20evaluation%20metrics.png" width=“200”>


