import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage
from matplotlib import pyplot as plt

def hierarchical_clustering():
    '''
    Erstellen eines Datensatzes wie in der Vorlesung
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
    Erstellen von Labels für jeden einzelnen Datensatz, was aber nur die Nummer ist,
    damit man dies besser ins dendogram überfuehren kann
    '''
    labels = range(1, len(X)+1)
    '''
    figsize= width / height in inches
    '''
    plt.figure(figsize=(10, 7))

    '''
    IST DAS WIRKLICH NOTWENDIG?
    '''
    plt.subplots_adjust(bottom=0.1)

    '''
    Daten anzeigen lassen
    X[:, 0] sind die X-Achsen Werte
    X[:, 1] sind die Y-Achsen Werte
    '''
    plt.scatter(X[:, 0], X[:, 1], label='True Position')

    '''
    Hinzufügen der Datenpunkte
    textcoords wird gesetzt, damit der Text nicht über dem Punkt steht
    '''
    for label, x, y in zip(labels, X[:, 0], X[:, 1]):
        plt.annotate(
            label,
            xy=(x, y), xytext=(-3, 3),
            textcoords='offset points', ha='right', va='bottom')
    plt.show()

    '''
    linkage führt die eigentliche Zusammenfügung der immer am naheliegendsten Datenpunkte durch
    single
    '''
    linked = linkage(X, 'single')

    plt.figure(figsize=(10, 7))

    '''
    orientation ist, wo der Knoten sich befindet
    distance_sort ist ein wenig der Aufbau, kann noch mit 'false' getestet werden
    '''
    dendrogram(linked,
                orientation='top',
                labels=labels,
                distance_sort='descending'
               )
    plt.show()

