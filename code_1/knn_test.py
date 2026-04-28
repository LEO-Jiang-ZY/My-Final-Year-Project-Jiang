import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

# 步骤1: 加载数据
fcdi_csv_path = r"D:\exploring_clouds\data\test20251019_match_label.csv"  # 替换为你的实际路径
df = pd.read_csv(fcdi_csv_path)

# 提取特征和标签
feature_cols = [col for col in df.columns if col.startswith('channel_')]  # 所有FCDI通道对
X = df[feature_cols]
y = df['label']

# 处理缺失值
X = X.fillna(0)  # 或 X = X.dropna()

# 分割数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 步骤2: 训练KNN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)
acc_knn = accuracy_score(y_test, y_pred_knn)
cm_knn = confusion_matrix(y_test, y_pred_knn)

# 训练随机森林
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
acc_rf = accuracy_score(y_test, y_pred_rf)
cm_rf = confusion_matrix(y_test, y_pred_rf)

# 步骤3: 可视化热力图
labels = sorted(df['label'].unique())  # 标签如[0,1,2,3,4,5]

plt.figure(figsize=(12, 5))

# KNN热力图
plt.subplot(1, 2, 1)
sns.heatmap(cm_knn, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.title(f'KNN Confusion Matrix (Accuracy: {acc_knn:.2f})')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')

# 随机森林热力图
plt.subplot(1, 2, 2)
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.title(f'Random Forest Confusion Matrix (Accuracy: {acc_rf:.2f})')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')

plt.tight_layout()
plt.savefig('classification_heatmap.png')  # 保存热力图
plt.show()  # 或 plt.close() 如果不需要显示

print(f"KNN Accuracy: {acc_knn:.2f}")
print(f"Random Forest Accuracy: {acc_rf:.2f}")
print("热力图已保存为 'classification_heatmap.png'")