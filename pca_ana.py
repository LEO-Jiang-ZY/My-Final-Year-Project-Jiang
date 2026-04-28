import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import StandardScaler
import random

# ============ 1. 读取与采样 ============
file_path = r"D:\exploring_clouds\data\test20251019.csv"   # TODO: 替换为你的 FCDI/CrIS 数据路径
df = pd.read_csv(file_path)

# 假设列结构: ['lat', 'lon', 'band1', 'band2', ..., 'bandN', 'label']
feature_cols = [c for c in df.columns if c not in ['lat', 'lon', 'label']]

# 随机抽取1000行
sample_df = df.sample(n=1000, random_state=42)
X = sample_df[feature_cols].values
y = sample_df['label'].values

# ============ 2. 数据标准化 ============
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ============ 3. PCA降维可视化 ============
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(8,6))
scatter = plt.scatter(X_pca[:,0], X_pca[:,1], c=y, cmap='viridis', s=20)
plt.colorbar(scatter, label='Cloud Label')
plt.title("PCA Visualization of CrIS Spectral Data")
plt.xlabel("PC1"); plt.ylabel("PC2")
plt.show()

# ============ 4. KNN 分类 ============
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

print("=== KNN Classification Report ===")
print(classification_report(y_test, y_pred))
print("\n=== Confusion Matrix ===")
print(confusion_matrix(y_test, y_pred))

# ============ 5. ROC 曲线 (仅二分类时) ============
if len(np.unique(y)) == 2:
    y_prob = knn.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label=f'KNN ROC curve (AUC={roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.show()

# ============ 6. 保存采样行号与经纬度 ============
out_path = 'sample_info.txt'
sample_info = sample_df[['lat', 'lon']].copy()
sample_info['row_index'] = sample_df.index
sample_info.to_csv(out_path, index=False, sep='\t')
print(f"Sample info saved to {out_path}")
