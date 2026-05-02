import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import f1_score, roc_auc_score, roc_curve

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

# ======================
# LOAD DATASET
# ======================
data = load_breast_cancer()
X = data.data
y = data.target

# ======================
# MODEL
# ======================
models = {
    "Logistic Regression": LogisticRegression(max_iter=5000),
    "Random Forest": RandomForestClassifier(random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5)
}

cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

hasil = []

plt.figure(figsize=(8,6))

for name, model in models.items():

    # prediksi probabilitas cross validation
    y_proba = cross_val_predict(
        model, X, y, cv=cv, method="predict_proba"
    )[:,1]

    y_pred = (y_proba >= 0.5).astype(int)

    acc = accuracy_score(y, y_pred)
    pre = precision_score(y, y_pred)
    rec = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    auc = roc_auc_score(y, y_proba)

    hasil.append({
        "Model": name,
        "Accuracy": round(acc,3),
        "Precision": round(pre,3),
        "Recall": round(rec,3),
        "F1 Score": round(f1,3),
        "ROC AUC": round(auc,3)
    })

    fpr, tpr, _ = roc_curve(y, y_proba)
    plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")

# garis random
plt.plot([0,1],[0,1],'k--')

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend()
plt.grid(True)
plt.show()

df = pd.DataFrame(hasil)
print(df)