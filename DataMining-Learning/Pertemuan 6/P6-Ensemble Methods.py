import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

# =====================
# LOAD DATASET
# =====================
data = load_breast_cancer()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# =====================
# MODEL
# =====================

models = {
    "Decision Tree":
        DecisionTreeClassifier(random_state=42),

    "Random Forest":
        RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=42
        ),

    "Gradient Boosting":
        GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            random_state=42
        )
}

# Stacking
base_models = [
    ('rf', RandomForestClassifier(n_estimators=50, random_state=42)),
    ('knn', KNeighborsClassifier(n_neighbors=5)),
    ('dt', DecisionTreeClassifier(max_depth=5, random_state=42))
]

stacking = StackingClassifier(
    estimators=base_models,
    final_estimator=LogisticRegression(max_iter=1000)
)

models["Stacking"] = stacking

# =====================
# EVALUASI
# =====================
hasil = []

for nama, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    hasil.append({
        "Model": nama,
        "Accuracy": round(accuracy_score(y_test, pred),3),
        "Precision": round(precision_score(y_test, pred),3),
        "Recall": round(recall_score(y_test, pred),3),
        "F1 Score": round(f1_score(y_test, pred),3)
    })

df = pd.DataFrame(hasil)
print(df)