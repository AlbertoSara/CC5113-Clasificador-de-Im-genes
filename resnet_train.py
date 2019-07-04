# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 02:53:51 2019

@author: Alberto
"""

import time
import numpy as np
from sklearn import cluster
from sklearn.decomposition import PCA
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
import glob
from random import shuffle
from joblib import dump


training_size = 3200

print("Process start")
start_total = time.time()
names = glob.glob("imagenes proyecto/*.jpg")
shuffle(names)
names = names[:training_size]
start = time.time()
model = ResNet50(weights='imagenet', include_top = False)
end = time.time()
time_taken = end - start
print('ResNet50 loading time: ',time_taken, " seconds")
all_images = np.zeros((len(names),224,224,3))


start = time.time()
for i, img_path in enumerate(names):
    img = image.load_img(img_path, target_size=(224, 224))
    img2 =  np.expand_dims(image.img_to_array(img),axis=0)
    all_images[i,:,:,:] = img2
end = time.time()
time_taken = end - start
print('Image loading time: ',time_taken, " seconds")
    
start = time.time()
all_images = preprocess_input(all_images)
time_taken = end - start
print('Preprocess time: ',time_taken, " seconds")

start = time.time()
predictions = model.predict(all_images)
del all_images
end = time.time()
time_taken = end - start
print('Feature extraction time: ',time_taken, " seconds")
predictions = predictions.reshape((predictions.shape[0],-1))
pca_comp = 62
start = time.time()
pca = PCA(n_components = pca_comp)
pca_predictions = pca.fit_transform(predictions)
end = time.time()
time_taken = end - start

print('PCA fit-transform time: ',time_taken, " seconds")
start = time.time()
kmeans = cluster.KMeans(n_clusters = 170, n_init=20, n_jobs = 4, verbose = 1)
pred_kmeans = kmeans.fit_predict(pca_predictions)
end = time.time()
time_taken = end - start

print('Cluster fit-predict time: ',time_taken, " seconds")
end_total = time.time()
time_taken = end_total - start_total
print('Total time: ',time_taken, " seconds")
dump(pca, "models/pca.joblib")
dump(kmeans, "models/kmeans.joblib")