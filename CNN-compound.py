import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import vgg16
from keras.applications.vgg16 import preprocess_input
from keras.utils.np_utils import to_categorical
from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.models import Sequential
from keras import optimizers
import numpy as np


#---- vars
num_compunds = 5 #classes
image_size = 224
batch_size = 32
data_src = '/pacbed-results' #pacbed data
cmpnd_folders = ['SrS/', 'PbS/', 'Sr3PbS/', 'SrPb3S/', 'SrPbS2/']

#rescale image between min and max
def scale_range (input, min, max):
    input += -(np.min(input))
    input /= np.max(input) / (max - min)
    input += min
    return input  


#------ data stuff
compounds = ['SrS', 'PbS', 'Sr3PbS', 'SrPb3S', 'SrPbS2']
#      labels  0      1       2          3         4
data_images = []
data_labels = []

#--- prepare labels and data
for c in range(5): #compunds
    path = data_src + cmpnd_folders[c]
    for i in range(200): #images
        img = np.load(path + 'pacbed-'+str(i)+compounds[c])
        img = scale_range(img,0,1)#image scaled between 0,1 
        img = img.astype(dtype=np.float32)
        img_size = img.shape[0]
        sx, sy = img.shape[0], img.shape[1]
        new_channel = np.zeros((img_size, img_size))
        img_stack = np.dstack((img, new_channel, new_channel))

        data_images.append(img_stack)
        data_labels.append(c)

nb_train_samples = len(data_images)
nb_class = len(set(data_labels))
x_train = np.concatenate([arr[np.newaxis] for arr in data_images])
y_train = to_categorical(data_labels, num_classes=nb_class)
print('Size of image array in bytes')
print(x_train.nbytes)

#------- Model
#vgg, frozen layers to function as feature extractor
input_shape = (data_images[0].shape[0],data_images[0].shape[1],3)
model = Sequential()

model.add(Conv2D(16, kernel_size=(3, 3), activation='relu', 
                 input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dense(1, activation='sigmoid'))


model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(),
              metrics=['accuracy'])


history = model.fit(x=x_train, y=y_train,
                    validation_data=(validation_imgs_scaled, validation_labels_enc),
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1)



vgg = vgg16.VGG16(include_top=False, weights='imagenet', 
                                     input_shape=input_shape)
output = vgg.layers[-1].output
output = keras.layers.Flatten()(output)
vgg_model = Model(vgg.input, output)

vgg_model.trainable = False
for layer in vgg_model.layers:
    layer.trainable = False

data_generator = ImageDataGenerator(preprocessing_function = preprocess_input)
train_generator=data_generator.flow_from_directory(
    data_src,
    target_size=(image_size, image_size),
    batch_size=batch_size,
    class_mode = 'categorical'
    )

validation_generator=data_generator.flow_from_directory(
    data_src,
    target_size=(image_size, image_size),
    batch_size=batch_size,
    class_mode = 'categorical'
    )

model = Sequential()
model.add(
    vgg16(include_top=False, pooling='avg', weights='imagenet')
)

#--- manipulate images
datagen = ImageDataGenerator(
        featurewise_center=True,
        rotation_range=90,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=1,
        vertical_flip=1,
        shear_range=0.05)