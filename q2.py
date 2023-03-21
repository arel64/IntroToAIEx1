import numpy as np
from sklearn.datasets import fetch_openml




def knn_algorithm(image_set, image_labels,query_image,k):
    distance_vec = []
    for image_vector in image_set:
        distance_vec += [np.linalg.norm(image_vector - query_image)]
    
    idx = np.argpartition(distance_vec, k)
    k_closest_labels = image_labels[idx[:k]]
    print(k_closest_labels)

     



mnist = fetch_openml("mnist_784", as_frame=False)
data = mnist["data"]
labels = mnist["target"]

size = 11000
range  = 70000
cutoff = 10000
idx = np.random.RandomState(0).choice(range, size)
train = data[idx[:cutoff], :].astype(int)
train_labels = labels[idx[:cutoff]]
test = data[idx[cutoff:], :].astype(int)
test_labels = labels[idx[cutoff:]]
knn_algorithm(train,train_labels,test[0],5)


