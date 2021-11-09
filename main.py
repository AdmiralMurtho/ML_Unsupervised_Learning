
from blob_clustering import blob_clustering
from hierarchical_clustering import hierarchical_clustering
from hierarchical_clustering_cluster import hierarchical_clustering_cluster
from elbow import find_k

if __name__ == '__main__':
    '''
    mit blob_clustering() wird ein kmean durchgefehrt.
    '''
    # blob_clustering()

    '''
    hierarchical_clustering() fuehrt ein Hierarchical Clustering aus
    '''
    # hierarchical_clustering()

    '''
    hierarchical_clustering_cluster() zeigt das mit hierarchical geclusterte Ergebnis
    mit eingefaerbten Punkten dar
    '''
    # hierarchical_clustering_cluster()

    '''
    elbow() fuehrt eine Elbow-Method aus, benoetigt aber die Daten uebergeben und
    zeigt dann das Plot an
    '''
    # find_k(points)


