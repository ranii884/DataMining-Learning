# Pertemuan 3
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from scipy.spatial.distance import euclidean, cityblock, minkowski
from sklearn.preprocessing import StandardScaler

# Load dataset Iris
iris = load_iris()
X = iris.data
df = pd.DataFrame(X, columns=iris.feature_names)

# Normalisasi data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Ambil 2 data pertama untuk contoh perhitungan jarak
A = X_scaled[0]
B = X_scaled[1]

print("=== PERHITUNGAN JARAK ===")
print("Euclidean :", round(euclidean(A,B),3))
print("Manhattan :", round(cityblock(A,B),3))
print("Minkowski (p=3):", round(minkowski(A,B,3),3))

# Clustering KMeans
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X_scaled)

df["Cluster"] = kmeans.labels_

print("\n=== HASIL CLUSTERING ===")
print(df.head(10))