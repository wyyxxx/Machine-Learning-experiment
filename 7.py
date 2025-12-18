import time
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier


# 1. 加载数据集
df = pd.read_csv('iris_data/iris.data', header=None)

X = df.iloc[:, :4].values
y = df.iloc[:, 4].values

target_names = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']

# 划分训练集/测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 2. 支持向量机
svm_model = SVC(kernel='rbf', C=1.0, gamma='scale')

start_train = time.time()
svm_model.fit(X_train_scaled, y_train)
svm_train_time = time.time() - start_train

start_pred = time.time()
y_pred_svm = svm_model.predict(X_test_scaled)
svm_pred_time = time.time() - start_pred

svm_accuracy = accuracy_score(y_test, y_pred_svm)

print("=== SVM 分类器 结果 ===")
print(f"训练时间 (s): {svm_train_time:.5f}")
print(f"预测时间 (s): {svm_pred_time:.5f}")
print(f"准确率: {svm_accuracy:.4f}")
print(classification_report(y_test, y_pred_svm, target_names=target_names))


# 3. 神经网络
mlp_model = MLPClassifier(
    hidden_layer_sizes=(50,),
    activation='relu',
    solver='adam',
    max_iter=500,
    random_state=42
)

start_train = time.time()
mlp_model.fit(X_train_scaled, y_train)
mlp_train_time = time.time() - start_train

start_pred = time.time()
y_pred_mlp = mlp_model.predict(X_test_scaled)
mlp_pred_time = time.time() - start_pred

mlp_accuracy = accuracy_score(y_test, y_pred_mlp)

print("=== 神经网络 MLP 分类器 结果 ===")
print(f"训练时间 (s): {mlp_train_time:.5f}")
print(f"预测时间 (s): {mlp_pred_time:.5f}")
print(f"准确率: {mlp_accuracy:.4f}")
print(classification_report(y_test, y_pred_mlp, target_names=target_names))

