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