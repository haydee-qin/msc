
from sklearn.cluster import KMeans, SpectralClustering as SklearnSpectralClustering
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
import numpy as np
from scipy.spatial.distance import pdist, squareform

def kmeans(X, k, max_iter=300, tol=1e-4, random_state=None):
    np.random.seed(random_state)
    centroids = X[np.random.choice(X.shape[0], k, replace=False)]

    for _ in range(max_iter):
        distances = cdist(X, centroids, 'euclidean')
        labels = np.argmin(distances, axis=1)

        new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(k)])

        if np.linalg.norm(new_centroids - centroids) < tol:
            break

        centroids = new_centroids

    return labels, centroids





def affinity_matrix(X, n_neighbors=10):
    pairwise_dists = squareform(pdist(X, 'euclidean'))
    sorted_dists = np.sort(pairwise_dists, axis=1)[:, 1:n_neighbors + 1]
    idx = np.argsort(pairwise_dists)[:, 1:n_neighbors + 1]

    A = np.zeros_like(pairwise_dists)
    for i in range(X.shape[0]):
        A[i, idx[i]] = sorted_dists[i]
        A[idx[i], i] = sorted_dists[i]

    return A


def normalized_laplacian(A):
    D = np.diag(1 / np.sqrt(np.sum(A, axis=1)))
    L = np.identity(A.shape[0]) - D @ A @ D
    return L


def spectral_clustering(X, k, n_neighbors=10, random_state=None):
    A = affinity_matrix(X, n_neighbors)
    L = normalized_laplacian(A)

    np.random.seed(random_state)
    _, eigvecs = np.linalg.eigh(L)

    Y = eigvecs[:, 1:k + 1]
    Y = Y / np.linalg.norm(Y, axis=1)[:, np.newaxis]

    labels, _ = kmeans(Y, k)
    return labels


# Read the files
with open('vocab.kos.txt', 'r') as f:
    vocab = f.read().splitlines()

with open('docword.kos.txt', 'r') as f:
    data = f.readlines()[3:]
k_values = range(2, 11)
kmeans_results = []
# Create a sparse matrix from the data
rows, cols, data_sparse = [], [], []
for line in data:
    row, col, count = map(int, line.split())
    rows.append(row - 1)
    cols.append(col - 1)
    data_sparse.append(count)

X = np.zeros((max(rows) + 1, len(vocab)))
for i, row in enumerate(rows):
    X[row, cols[i]] = data_sparse[i]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)




# n_components sh  ould be less than the number of features in your dataset.
n_components = 10
svd = TruncatedSVD(n_components=n_components)
X_reduced = svd.fit_transform(X_scaled)


kmeans_custom_results = []
kmeans_sklearn_results = []
spectral_custom_results = []
spectral_sklearn_results = []

for k in k_values:
    # Custom k-means implementation
    labels_custom, _ = kmeans(X_reduced, k, random_state=42)
    silhouette_custom = silhouette_score(X_reduced, labels_custom)
    kmeans_custom_results.append((k, silhouette_custom))

    # Sklearn k-means implementation
    kmeans_sklearn = KMeans(n_clusters=k, random_state=42).fit(X_reduced)
    labels_sklearn = kmeans_sklearn.labels_
    silhouette_sklearn = silhouette_score(X_reduced, labels_sklearn)
    kmeans_sklearn_results.append((k, silhouette_sklearn))

    # Custom spectral clustering implementation
    spec_labels_custom = spectral_clustering(X_reduced, k)
    spec_silhouette_custom = silhouette_score(X_reduced, spec_labels_custom)
    spectral_custom_results.append((k, spec_silhouette_custom))

    # Sklearn spectral clustering implementation
    spectral_sklearn = SklearnSpectralClustering(n_clusters=k, affinity='nearest_neighbors').fit(X_reduced)
    spec_labels_sklearn = spectral_sklearn.labels_
    spec_silhouette_sklearn = silhouette_score(X_reduced, spec_labels_sklearn)
    spectral_sklearn_results.append((k, spec_silhouette_sklearn))

# Plot Silhouette scores for custom and sklearn implementations of k-means and spectral clustering
plt.figure(figsize=(12, 8))
plt.plot([result[0] for result in kmeans_custom_results], [result[1] for result in kmeans_custom_results], marker='o', label='Custom K-means')
plt.plot([result[0] for result in kmeans_sklearn_results], [result[1] for result in kmeans_sklearn_results], marker='s', label='Sklearn K-means')
plt.plot([result[0] for result in spectral_custom_results], [result[1] for result in spectral_custom_results], marker='x', label='Custom Spectral Clustering')
plt.plot([result[0] for result in spectral_sklearn_results], [result[1] for result in spectral_sklearn_results], marker='*', label='Sklearn Spectral Clustering')
plt.xlabel('Number of clusters')
plt.ylabel('Silhouette score')
plt.title('Silhouette Scores for Custom and Sklearn Implementations of K-means and Spectral Clustering Algorithms')
plt.legend()
plt.grid()
plt.show()