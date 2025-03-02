
"""


@author: Jacqueline Chiazor
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from urllib.request import u

# 1. Data Preprocessing
# Download the Iris dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
filename = "iris.data"
urlretrieve(url, filename)

# Load the Iris dataset
column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
iris_df = pd.read_csv(filename, header=None, names=column_names)

X = iris_df.iloc[:, :-1].values
y = iris_df.iloc[:, -1].values

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 2. Elbow Method to Determine Optimal k (Number of Clusters)
wcss = []  # List to store WCSS for different k values

# Loop through k values from 1 to 10 (adjust as needed)
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)  # Fit the model
    wcss.append(kmeans.inertia_)  # WCSS (inertia) of the current model

# Plot the WCSS values to visualize the "elbow"
plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), wcss, marker='o', linestyle='--')
plt.title('Elbow Method for Optimal k')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('WCSS')
plt.show()

# 3. K-Means Clustering with 3 Clusters (After Elbow Method)
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X_scaled)

# 4. Evaluation: Visualize the clusters
plt.figure(figsize=(12, 5))

# Plot for the first two features (sepal length vs. sepal width)
plt.subplot(121)
scatter = plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=kmeans.labels_, cmap='viridis')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
            marker='*', s=300, c='red', label='Centroids')
plt.title('K-Means Clustering on Iris Dataset\n(Sepal Length vs Sepal Width)')
plt.xlabel('Standardized Sepal Length')
plt.ylabel('Standardized Sepal Width')
plt.colorbar(scatter)
plt.legend()

# Plot for the last two features (petal length vs. petal width)
plt.subplot(122)
scatter = plt.scatter(X_scaled[:, 2], X_scaled[:, 3], c=kmeans.labels_, cmap='viridis')
plt.scatter(kmeans.cluster_centers_[:, 2], kmeans.cluster_centers_[:, 3], 
            marker='*', s=300, c='red', label='Centroids')
plt.title('K-Means Clustering on Iris Dataset\n(Petal Length vs Petal Width)')
plt.xlabel('Standardized Petal Length')
plt.ylabel('Standardized Petal Width')
plt.colorbar(scatter)
plt.legend()

plt.tight_layout()
plt.show()

# 5. Compare Clustering with Ground Truth
species_to_num = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
y_numeric = np.array([species_to_num[species] for species in y])
comparison = pd.crosstab(kmeans.labels_, y_numeric, rownames=['Clusters'], colnames=['Species'])
print("Cluster vs Species Comparison:")
print(comparison)

# 6. Calculate Silhouette Score
silhouette_avg = silhouette_score(X_scaled, kmeans.labels_)
print(f"\nSilhouette Score: {silhouette_avg:.3f}")

# 7. Cluster Purity Analysis
print("\nCluster Assignments and Species Distribution:")
for cluster in comparison.index:
    most_common_species = comparison.loc[cluster].idxmax()  # Species with the highest count in the cluster
    correct_count = comparison.loc[cluster, most_common_species]
    total_count = comparison.loc[cluster].sum()
    accuracy = correct_count / total_count
    print(f"Cluster {cluster}: Most corresponds to Species {most_common_species} with accuracy {accuracy:.2f} ({correct_count}/{total_count})")

# 8. Overall Cluster Purity
total_correct = sum(comparison.max(axis=1))  # Sum of the maximum counts for each cluster
total_samples = comparison.values.sum()      # Total number of samples
purity = total_correct / total_samples
print(f"\nOverall Cluster Purity: {purity:.2f}")

# 9. Map Clusters to Species for Clarity
cluster_to_species = {cluster: comparison.loc[cluster].idxmax() for cluster in comparison.index}
mapped_labels = [cluster_to_species[label] for label in kmeans.labels_]

# Sample comparison of predicted vs. true species for 10 random samples
sample_indices = np.random.choice(len(y), 10, replace=False)  # Random sample of 10 points
print("\nSample Comparison of Predicted vs. True Species:")
for idx in sample_indices:
    print(f"Sample {idx}: Predicted = {mapped_labels[idx]}, True = {y[idx]}")

# 10. Cluster Centroid Analysis
centroids = kmeans.cluster_centers_
centroids_df = pd.DataFrame(centroids, columns=column_names[:-1])

print("\nCluster Centroids (Standardized Features):")
print(centroids_df)

# Transform centroids back to the original scale for interpretability
centroids_original_scale = scaler.inverse_transform(centroids)
centroids_original_df = pd.DataFrame(centroids_original_scale, columns=column_names[:-1])

print("\nCluster Centroids (Original Scale):")
print(centroids_original_df)
