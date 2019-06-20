# -*- coding: utf-8 -*-
"""
Created on Sun Jun 16 18:27:04 2019

@author: Alberto
"""


#Time:  3747.6734867095947  seconds
import time

import numpy as np
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint
from random import shuffle
import matplotlib.pyplot as plt
import cv2
import glob


img_size = 224      #tamaño de la imagen
batch = 50          #tamaño del batch para entrenar, mientras más alto mejor
epochs = 50         #epochs
filepath="tryhard224c.hdf5"#"models/tryhard224c.hdf5"      #donde queda guardado el modelo entrenado


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

input_img = layers.Input(shape=(img_size, img_size, 3))  

x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
x = layers.MaxPooling2D((2, 2), padding='same')(x)
x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(x)
x = layers.MaxPooling2D((2, 2), padding='same')(x)
x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
encoded = layers.MaxPooling2D((2, 2), padding='same')(x)


x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
x = layers.UpSampling2D((2, 2))(x)
x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(x)
x = layers.UpSampling2D((2, 2))(x)
x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = layers.UpSampling2D((2, 2))(x)
decoded = layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)


autoencoder = models.Model(input_img, decoded)
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy', metrics=['accuracy'])

checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]
#   
start = time.time()
history = autoencoder.fit(img, img, batch_size = batch, epochs=epochs, validation_split = 0.2, callbacks=callbacks_list, shuffle=True)
end = time.time()
time_taken = end - start
print('Time: ',time_taken, " seconds")

plt.figure()
accuracy = history.history['acc']
val_accuracy = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(accuracy))
plt.plot(epochs, accuracy, label='Training accuracy')
plt.plot(epochs, val_accuracy, label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, label='Training loss')
plt.plot(epochs, val_loss, label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()
