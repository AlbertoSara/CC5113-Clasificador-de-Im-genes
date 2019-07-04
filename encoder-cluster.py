# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 01:48:35 2019

@author: Alberto
"""
#Time:  99.2205491065979  seconds

import time
import numpy as np
import cv2
from tensorflow.keras import models
from sklearn import cluster
from sklearn.decomposition import PCA
from sklearn import mixture
import matplotlib.pyplot as plt
from random import shuffle
import glob




img_size = 224
pca_comp = 2000  #componentes de pca, va entre 0 y min(num. datos, features)
                #numero de features = 6272
names = glob.glob("imagenes proyecto/*.jpg")
shuffle(names)
shuffle(names)

img = []
nms = []
shuffle(names) 
print("Loading pictures")
for name in names:
    try:
        img.append(cv2.resize(cv2.imread(name)/255.0, (img_size, img_size)))
        nms.append(name)
    except KeyboardInterrupt:
        break
    except:
        pass
img = np.asarray(img)

autoencoder = models.load_model('models/tryhard224c.hdf5')
encoder = models.Sequential()
for i in range(len(autoencoder.layers)):
    if i < 7:
        encoder.add(autoencoder.layers[i])
    else:
        break
    
encoder.summary()
predictions = encoder.predict(img)
predictions = predictions.reshape((predictions.shape[0],-1))
pca = PCA(n_components = pca_comp)
pca_predictions = pca.fit_transform(predictions)
pca_2 = PCA(n_components = 2)
pca_small = pca_2.fit_transform(predictions)
start = time.time()
kmeans = cluster.KMeans(n_clusters = 170, n_init=20, n_jobs = 4, verbose = 1)
pred_kmeans = kmeans.fit_predict(pca_predictions)
end = time.time()
time_taken = end - start

print('Time: ',time_taken, " seconds")

fig, axs = plt.subplots(nrows=2, ncols=5, figsize=(25, 23),
                        subplot_kw={'xticks': [], 'yticks': []})

i = 0
for ax in axs.flat:
    ax.imshow(img[i,:,:,:])
    ax.set_title(pred_kmeans[i])
    i = i + 1


plt.figure()
plt.scatter(pca_small[:,0],pca_small[:,1],c=pred_kmeans)
plt.show()

plt.show()