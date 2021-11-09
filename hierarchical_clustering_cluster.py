import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import AgglomerativeClustering

def hierarchical_clustering_cluster():
    linkage = 'single' #'complete' / 'average'

    '''fd
    X ist der Datensatz, mit dem das Clustering durchgefuehrt wird 
    '''
    X = np.array([[5, 3],
                  [10, 15],
                  [15, 12],
                  [24, 10],
                  [30, 30],
                  [85, 70],
                  [71, 80],
                  [60, 78],
                  [70, 55],
                  [80, 91], ])

    '''
    Die Angabe der Cluster wird hier durch n_clusters angegeben und das linkage
    kann oben bestimmt werden. 
    '''
    cluster = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage=linkage)
    cluster.fit_predict(X)

    plt.scatter(X[:, 0], X[:, 1], c=cluster.labels_, cmap='rainbow')
    plt.show()
