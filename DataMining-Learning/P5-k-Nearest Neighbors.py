# Pertemuan 5
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from collections import Counter

# ======================
# LOAD DATASET
# ======================
iris = load_iris()
X = iris.data
y = iris.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Normalisasi
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ======================
# KNN SCRATCH
# ======================
def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

def knn_predict(X_train, y_train, x_test, k):
    distances = []

    for i in range(len(X_train)):
        dist = euclidean_distance(X_train[i], x_test)
        distances.append((dist, y_train[i]))

    distances.sort(key=lambda x: x[0])
    neighbors = distances[:k]

    labels = [label for _, label in neighbors]
    prediction = Counter(labels).most_common(1)[0][0]

    return prediction

# ======================
# UJI BERBAGAI NILAI K
# ======================
k_values = [1,3,5,7,9,11]

print("=== PERBANDINGAN AKURASI ===")

for k in k_values:

    # Scratch
    y_pred_scratch = []
    for x in X_test:
        pred = knn_predict(X_train, y_train, x, k)
        y_pred_scratch.append(pred)

    acc_scratch = accuracy_score(y_test, y_pred_scratch)

    # Sklearn
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    y_pred_sklearn = model.predict(X_test)

    acc_sklearn = accuracy_score(y_test, y_pred_sklearn)

    print(f"k={k}")
    print("Scratch :", round(acc_scratch,3))
    print("Sklearn :", round(acc_sklearn,3))
    print("-"*30)