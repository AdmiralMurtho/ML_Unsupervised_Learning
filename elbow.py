import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

"""
Mit find_k wird anhand der gegebenen Punkte ein Plott ausgegeben, aus dem man dann
den Wert des am besten passenden K auslesen
"""
def find_k(points):
    distortions = []
    for i in range(1,11):
        km = KMeans(n_clusters=i, init='random', n_init=10, max_iter=300, tol=1e-04, random_state=0)
        km.fit(points)
        distortions.append(km.inertia_)

    plt.plot(range(1,11), distortions, marker='o')
    plt.xlabel('Number of clusters')
    plt.ylabel('Distortions')
    plt.show()


    return 3