# Pertemuan 4
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# ===============================
# LOAD DATASET IRIS
# ===============================
iris = load_iris()
X = iris.data[:, :2]   # ambil 2 fitur biar sederhana
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# ===============================
# DECISION TREE SEDERHANA (SCRATCH)
# aturan manual berdasarkan split sederhana
# ===============================
def simple_tree_predict(x):
    # split pakai feature pertama (sepal length)
    if x[0] < 5.5:
        return 0
    elif x[0] < 6.5:
        return 1
    else:
        return 2

# prediksi scratch
y_pred_scratch = np.array([simple_tree_predict(x) for x in X_test])

# ===============================
# DECISION TREE SCIKIT-LEARN
# ===============================
model = DecisionTreeClassifier(max_depth=3, random_state=42)
model.fit(X_train, y_train)

y_pred_sklearn = model.predict(X_test)

# ===============================
# EVALUASI
# ===============================
acc_scratch = accuracy_score(y_test, y_pred_scratch)
acc_sklearn = accuracy_score(y_test, y_pred_sklearn)

print("=== HASIL PERBANDINGAN ===")
print("Akurasi Decision Tree Scratch :", round(acc_scratch,3))
print("Akurasi Decision Tree Sklearn :", round(acc_sklearn,3))