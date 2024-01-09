import os
import sys
from keras.utils.np_utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras import applications
from keras import callbacks
from keras import optimizers
from keras.callbacks import EarlyStopping
from keras import backend as K
import tensorflow as tf
import numpy as np
import scipy.io as sio
import time

#---- vars
num_compunds = 100 #thickness from 0.5 nm to 100
image_size = 224
batch_size = 32
data_src = 'pacbed-results/' #pacbed data

data_images = []
data_labels = []
compound_folder = 'SrS/'


#rescale image between min and max
def scale_range (input, min, max):
    input += -(np.min(input))
    input /= np.max(input) / (max - min)
    input += min
    return input  


#--- data 
l = 0
for t in range(200): #thickness
    path = data_src + compound_folder + 'pacbed-' + str(t) + '-SrS'
    img = np.load(path)
    img = scale_range(img,0,1)#image scaled between 0,1 
    img = img.astype(dtype=np.float32)
    img_size = img.shape[0]
    sx, sy = img.shape[0], img.shape[1]
    new_channel = np.zeros((img_size, img_size))
    img_stack = np.dstack((img, new_channel, new_channel))

    data_images.append(img_stack)
    data_labels.append(t%2)

nb_train_samples = len(data_images)
nb_class = len(set(data_labels))
x_train = np.concatenate([arr[np.newaxis] for arr in data_images])
y_train = to_categorical(data_labels, num_classes=nb_class)
print('Size of image array in bytes')
print(x_train.nbytes)

#------- Model
#vgg, frozen layers to function as feature extractor
model = applications.VGG16(include_top=False, weights='imagenet')
datagen = ImageDataGenerator(
        featurewise_center=True,
        rotation_range=90,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=1,
        vertical_flip=1,
        shear_range=0.05)
 

datagen = ImageDataGenerator(
        featurewise_center=True)

datagen.fit(x_train)
print('made it past featurewise center')
generator = datagen.flow(
        x_train,
        y_train,
        batch_size=batch_size,
        shuffle=False)
print('made it past generator')

bottleneck_features_train = model.predict_generator(
        generator, nb_train_samples // batch_size)


model.add(Conv2D(16, kernel_size=(3, 3), activation='relu', 
                 input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(1, activation='sigmoid'))


model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(),
              metrics=['accuracy'])
              
              
history = model.fit(x=train_imgs_scaled, y=train_labels_enc,
                    validation_data=(validation_imgs_scaled, validation_labels_enc),
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1)     