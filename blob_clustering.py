import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from elbow import find_k



def blob_clustering():
    k_known = True
    K = 2
    _EXAMPLE_CENTERS = 2

    '''
    Erstellen der Blobs
    n_samples = anzahl der Datenpunkte
    centers = anzahl cluster
    cluster_std = Standarddeviation -> Je größer, desto näher sind sie beieinander
    random_state = Es wird ein bestimmter State genommen und nicht 
                bei jedem Durchlauf zufällig generiert
    '''
    dataset = make_blobs(
        n_samples=200,
        centers=_EXAMPLE_CENTERS,
        cluster_std=1.6,
        random_state=50,
    )

    '''
    Da in dataset[1] die tatsächlichen "Labels" stehen
    '''
    points = dataset[0]

    '''
    Anzeigen der Punkte
    '''
    plt.scatter(dataset[0][:, 0], dataset[0][:, 1])
    plt.show()

    '''
    Elbow-Method, kommt später
    '''
    if not k_known:
        find_k(points)
        K = input("Bitte gebe das gewünschte k ein: ")

    '''
    Erstellen einer kmeans instanz
    n_clusters = Anzahl der Cluster, die kmeans herausfinden soll
    n_init = default10, anzahl der Durchläufe mit unterschiedlichen Startpunkten.
            Der beste daraus resultierende Wert wird genommen.
    '''
    kmeans = KMeans(n_clusters=K, n_init=1)

    '''
    kmeans.fit heißt, dass kmeans durchgeführt wird
    '''
    kmeans.fit(points)

    '''
    Die koordinaten der Cluster-Centers
    '''
    clusters = kmeans.cluster_centers_

    '''
    es wird ein Array erstellt, wobei für jeden Datensatz die Zuordnung zu dem jeweiligen 
    Center gesetzt wird.
    '''
    y_km = kmeans.predict(points)

    # Ausgaben der Cluster
    for i in range(K):
        plt.scatter(points[y_km == i, 0], points[y_km == i, 1], s=50)
        plt.scatter(clusters[i][0], clusters[i][1], marker='*', s=100, color='black')
    plt.show()
