import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from elbow import find_k

k_known = True
K = 4
_EXAMPLE_CENTERS = 4

if __name__ == '__main__':

    dataset = make_blobs(
        n_samples=200,
        centers=_EXAMPLE_CENTERS,
        n_features=2,
        cluster_std=1.6,
        random_state=50)
    points = dataset[0]
    if not k_known:
        find_k(points)
        K = input("Bitte gebe das gewünschte k ein: ")

    kmeans = KMeans(n_clusters=K)
    kmeans.fit(points)
    plt.scatter(dataset[0][:, 0], dataset[0][:, 1])
    plt.show()
    clusters = kmeans.cluster_centers_
    y_km = kmeans.fit_predict(points)

    # Ausgaben der Cluster
    for i in range(K):
        plt.scatter(points[y_km == i, 0], points[y_km == i, 1], s=50)
        plt.scatter(clusters[i][0], clusters[i][1], marker='*', s=100, color='black')
    plt.show()

