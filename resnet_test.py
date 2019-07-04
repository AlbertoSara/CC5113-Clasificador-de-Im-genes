# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 02:53:51 2019

@author: Alberto
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
import glob
from joblib import load
import csv
    


print("Process start")
start_total = time.time()
class_dict = {}
with open('class.csv', 'r', encoding="UTF-8") as csvfile:
    reader = csv.reader(csvfile, delimiter=';')
    for row in reader:
        class_dict[int(row[0])] = row[1]
        

names = glob.glob("imagenes clasificar/*.jpg")
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
end = time.time()
time_taken = end - start
print('Feature extraction time: ',time_taken, " seconds")
predictions = predictions.reshape((predictions.shape[0],-1))
start = time.time()
pca = load("models/pca.joblib")
pca_predictions= pca.transform(predictions)

end = time.time()
time_taken = end - start

print('PCA transform time: ',time_taken, " seconds")
start = time.time()
kmeans = load("models/kmeans.joblib")
pred_kmeans = kmeans.predict(pca_predictions)

end = time.time()
time_taken = end - start

print('Cluster predict time: ',time_taken, " seconds")
end_total = time.time()
time_taken = end_total - start_total
print('Total time: ',time_taken, " seconds")
    
classified = {}
for i in range(len(names)):
    classified[names[i]] = class_dict[pred_kmeans[i]]
    


fig, axs = plt.subplots(nrows=3, ncols=5, figsize=(70, 70),
                    subplot_kw={'xticks': [], 'yticks': []})
i = 0
classified_list = list(classified.items())
for ax in axs.flat:
    img_name, img_class = classified_list[i]
    ax.imshow(image.load_img(img_name, target_size=(224, 224)))
    ax.set_title(img_class)
    i = i + 1
        
plt.show()

for i in classified_list:
    img_name, img_class = i
    print("Image: " + str(img_name) + " Class: " + str(img_class))