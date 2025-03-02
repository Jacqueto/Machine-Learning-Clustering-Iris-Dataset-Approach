# K-Means Clustering on the Iris Dataset

## Project Overview
This project explores the effectiveness of the **K-Means Clustering** algorithm on the **Iris dataset**. It applies unsupervised learning techniques to group data points into clusters based on flower characteristics.

## Dataset
- **Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data)
- **Features:**
  - Sepal length
  - Sepal width
  - Petal length
  - Petal width

## Implementation Details
- **Tools Used:**
  - Python (Google Colab)
  - Pandas, NumPy, Matplotlib
  - Scikit-learn (for K-Means Clustering)
- **Evaluation Metrics:**
  - Scatter Plots
  - Silhouette Score
  - Elbow Method
  - Cluster Centroids Analysis

## Results & Conclusion
- The Elbow Method suggested **3 clusters** for optimal classification.
- The **Silhouette Score (~0.6)** indicated moderate clustering performance.
- **Petal dimensions** provided clearer cluster separation than sepals.
- Some overlap between species suggests that alternative methods like **DBSCAN** or **Gaussian Mixture Models** might perform better.

## How to Run
1. Clone this repository:
   ```sh
   git clone <https://github.com/Jacqueto/Machine-Learning-Clustering-Iris-Dataset-Approach>
