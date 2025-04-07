from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import numpy as np


def extract_features(corpus, feature_type='tf'):
    """
    提取文本特征，支持两种特征提取方式：
    - 'tf': 高频词特征（词频）
    - 'tfidf': TF-IDF加权特征
    :param corpus: 文本数据列表
    :param feature_type: 特征提取方式，可选 'tf' 或 'tfidf'
    :return: 特征矩阵和特征名称
    """
    if feature_type == 'tf':
        # 使用词频（TF）作为特征
        vectorizer = CountVectorizer()
        X = vectorizer.fit_transform(corpus)
        feature_names = vectorizer.get_feature_names_out()
    elif feature_type == 'tfidf':
        # 使用TF-IDF作为特征
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(corpus)
        feature_names = vectorizer.get_feature_names_out()
    else:
        raise ValueError("Unsupported feature type. Choose 'tf' or 'tfidf'.")

    return X, feature_names


# 示例数据
corpus = [
    "This is the first document.",
    "This document is the second document.",
    "And this is the third one.",
    "Is this the first document?"
]

# 提取特征
feature_type = 'tfidf'  # 可以切换为 'tfidf'
print(f"Feature type set to: {feature_type}")  # 添加调试信息
X, feature_names = extract_features(corpus, feature_type)

# 打印特征矩阵和特征名称
print("Feature Names:", feature_names)
print("Feature Matrix:\n", X.toarray())