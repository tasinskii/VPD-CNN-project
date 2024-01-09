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

#-------------DATA
#path vars
path_main = 'Datasets/' #FIX for quest
cmpnd_folders = ['SrS/', 'PbS/', 'Sr3PbS/', 'SrPb3S/', 'SrPbS2/']

#rescale image between min and max
def scale_range (input, min, max):
    input += -(np.min(input))
    input /= np.max(input) / (max - min)
    input += min
    return input  
data_list = []
label_list = []
#loop thru files, collate data into training list

for c in range(0,5): #compounds
    for t in range(0,150): #thickness 45 nm, 150 increments of 3 angstrom
        for i in range(0, 15): # 15 images per thickness
            path = path_main + cmpnd_folders[c] + str(i) + '_' + str(t)
            img = np.load(path)
            label = int(str(c) + str(t) + str(i)) #see notes
            img = scale_range(img,0,1)#image scaled between 0,1 
            img = img.astype(dtype=np.float32)
            #add 2 extra channels to be compatible with VGG16
            img_size = img.shape[0]
            sx, sy = img.shape[0], img.shape[1]
            new_channel = np.zeros((img_size, img_size))
            img_stack = np.dstack((img, new_channel, new_channel))
            data_list.append(img_stack)
            label_list.append(label)
#finalize data
nb_train_samples = len(data_list)
nb_class = len(set(label_list))

x_train = np.concatenate([arr[np.newaxis] for arr in data_list])
y_train = to_categorical(label_list, num_classes=nb_class)
print('Size of image array in bytes')
print(x_train.nbytes)

#---------MODEL
batch_size = 32
#transfer learning? pre train on imagenet dataset
model = applications.VGG16(include_top=False, weights='imagenet')
#randomly change data 
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
generator = datagen.flow(
        x_train,
        y_train,
        batch_size=batch_size,
        shuffle=False)
bottleneck_features_train = model.predict_generator(
        generator, nb_train_samples // batch_size)

#train top models
train_data = bottleneck_features_train
train_labels = y_train
print(train_data.shape, train_labels.shape)
model = Sequential()
model.add(Flatten(input_shape=train_data.shape[1:]))
model.add(Dropout(0.3))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(nb_class, activation='softmax'))

# compile setting:
lr = 0.001
decay = 1e-6
momentum = 0.9
optimizer = optimizers.SGD(lr=lr, decay=decay, momentum=momentum, nesterov=True)
loss = 'categorical_crossentropy'
model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    
#bottleneck_log = result_path + 'training_' + str(max_index) + '_bnfeature_log.csv'
#csv_logger_bnfeature = callbacks.CSVLogger(bottleneck_log)
earlystop = EarlyStopping(monitor='val_acc', min_delta=0.0001, patience=3, verbose=1, mode='auto')

model.fit(train_data,train_labels,epochs=epochs,batch_size=batch_size,shuffle=True,
            callbacks=[csv_logger_bnfeature, earlystop],verbose=2,validation_split=0.2)
#fine tune

model = applications.VGG16(weights='imagenet', include_top=False, input_shape=(sx, sy, 3))

top_model = Sequential()
top_model.add(Flatten(input_shape=model.output_shape[1:]))
top_model.add(Dropout(0.3))
top_model.add(Dense(256, activation='relu'))
top_model.add(Dropout(0.3))
top_model.add(Dense(52, activation='softmax'))

top_model.load_weights(result_path + 'bottleneck_fc_model.h5')

new_model = Sequential()
for l in model.layers:
        new_model.add(l)
new_model.add(top_model)

    # compile settings
lr = 0.0001
decay = 1e-6
momentum = 0.9
optimizer = optimizers.SGD(lr=lr, decay=decay, momentum=momentum, nesterov=True)
loss = 'categorical_crossentropy'
new_model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

earlystop = EarlyStopping(monitor='val_acc', min_delta=0.0001, patience=5, verbose=1, mode='auto')

datagen = ImageDataGenerator(
        featurewise_center=True,
        rotation_range=90,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=1,
        vertical_flip=1,
        shear_range=0.05)

datagen.fit(train_data)

generator = datagen.flow(
        train_data,
        train_labels,
        batch_size=batch_size,
        shuffle=True)

validation_generator = datagen.flow(
        train_data,
        train_labels,
        batch_size=batch_size,
        shuffle=True)

new_model.fit_generator(generator,epochs=epochs,steps_per_epoch=len(train_data) / 32,validation_data=validation_generator,validation_steps=(len(train_data)//5)//32,
            callbacks=[csv_logger_finetune, earlystop],verbose=2)

    #new_model.fit(train_data, train_labels, epochs=epochs, batch_size=batch_size, shuffle=True, validation_split=0.2,
                  #callbacks=[csv_logger_finetune, earlystop])


new_model.save(result_path + 'FinalModel.h5')  # save the final model for future loading and prediction
