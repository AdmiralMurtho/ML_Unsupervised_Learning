from keras.datasets import mnist
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import accuracy_score
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np
from skimage import color
from skimage import io



def mnist_kmean():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Checking the 'Type'

    print(type(x_train))
    print(type(x_test))
    print(type(y_train))
    print(type(y_test))

    print(x_train.shape)
    print(x_test.shape)
    print(y_train.shape)
    print(y_test.shape)

    plt.gray()  # B/W Images
    plt.figure(figsize=(10, 9))  # Adjusting figure size
    # Displaying a grid of 3x3 images
    for i in range(9):
        plt.subplot(3, 3, i + 1)
        plt.imshow(x_train[i])
    plt.show()
    for i in range(5):
        print(y_train[i])
    # Darstellen des min/max
    print(x_train.min())
    print(x_train.max())

    # Data normalisation:

    # Conversion to float

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    x_train = x_train/255.0
    x_test = x_test/255.0

    print(x_train.min())
    print(x_train.max())

    # Reshaping input data
    # So shaping from 3D (60000*28*28) -> (60000*784)
    X_train = x_train.reshape(len(x_train), -1)
    X_test = x_test.reshape(len(x_test), -1)

    # Checking the shape
    print(X_train.shape)
    print(X_test.shape)

    total_clusters = len(np.unique(y_test))
    # Initialize the K-Means model
    kmeans = MiniBatchKMeans(n_clusters=total_clusters)
    # Fitting the model to training set
    kmeans.fit(X_train)

    kmeans.labels_

    def retrieve_info(cluster_labels,y_train):
        '''
        Associates most probable label with each cluster in KMeans model
        returns: dictionary of clusters assigned to each label
        '''
        # Initializing
        reference_labels = {}
        # For loop to run through each label of cluster label
        for i in range(len(np.unique(kmeans.labels_))):
            index = np.where(cluster_labels == i, 1, 0)
            num = np.bincount(y_train[index == 1]).argmax()
            reference_labels[i] = num
        return reference_labels

    reference_labels = retrieve_info(kmeans.labels_, y_train)
    number_labels = np.random.rand(len(kmeans.labels_))
    for i in range(len(kmeans.labels_)):
        number_labels[i] = reference_labels[kmeans.labels_[i]]

    # Comparing Predicted values and Actual values
    print(number_labels[:20].astype('int'))
    print(y_train[:20])

    print(accuracy_score(number_labels, y_train))

    # Function to calculate metrics for the model
    def calculate_metrics(model, output):
        print('Number of clusters is {}'.format(model.n_clusters))
        print('Inertia: {}'.format(model.inertia_))
        print('Homogeneity: {}'.format(metrics.homogeneity_score(output, model.labels_)))


    # Dieser Bereich wird genutzt, um den Vergleich zu zeigen, dass 9 Cluster nicht zwigend das beste ist
    
    cluster_number = [10, 16, 36, 64, 144, 256]
    for i in cluster_number:
        total_clusters = len(np.unique(y_test))
        # Initialize the K-Means model
        kmeans = MiniBatchKMeans(n_clusters=i)
        # Fitting the model to training set
        kmeans.fit(X_train)
        # Calculating the metrics

        calculate_metrics(kmeans, y_train)
        # Calculating reference_labels
        reference_labels = retrieve_info(kmeans.labels_, y_train)
        # ‘number_labels’ is a list which denotes the number displayed in image
        number_labels = np.random.rand(len(kmeans.labels_))
        for i in range(len(kmeans.labels_)):
            number_labels[i] = reference_labels[kmeans.labels_[i]]

        print('Accuracy score: {}'.format(accuracy_score(number_labels, y_train)))
        print('\n')


    # Testing model on Testing set
    # Initialize the K-Means model
    kmeans = MiniBatchKMeans(n_clusters=256)
    # Fitting the model to testing set
    kmeans.fit(X_test)
    # Calculating the metrics
    calculate_metrics(kmeans, y_test)
    # Calculating the reference_labels
    reference_labels = retrieve_info(kmeans.labels_, y_test)
    # ‘number_labels’ is a list which denotes the number displayed in image
    number_labels = np.random.rand(len(kmeans.labels_))
    for i in range(len(kmeans.labels_)):
        number_labels[i] = reference_labels[kmeans.labels_[i]]

    print('Accuracy score: {}'.format(accuracy_score(number_labels, y_test)))
    print('\n')

    centroids = kmeans.cluster_centers_

    centroids.shape

    centroids = centroids.reshape(256,28,28)
    centroids = centroids * 255

    plt.figure(figsize=(10, 9))

    bottom = 0.35

    for i in range(16):
        plt.subplots_adjust(bottom)
        plt.subplot(4,4,i+1)
        plt.title('Number: {}'.format(reference_labels[i]), fontsize=17)
        plt.imshow(centroids[i])
    plt.show()






