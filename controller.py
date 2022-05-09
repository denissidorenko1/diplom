# Import some packages to use

# Import some packages to use
import sys
import tensorflow
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# To see our directory
import os
import random
import gc  # Gabage collector for cleaning deleted data from memory
import cv2
import numpy as np

# To see our directory
import os
import random
import gc  # Gabage collector for cleaning deleted data from memory
import genetic_algorithm, neural_fitness
train_dir = '/Users/denis/PycharmProjects/diplom/input/train'
test_dir = '/Users/denis/PycharmProjects/diplom/input/test'

train_dogs = ['/Users/denis/PycharmProjects/diplom/input/train/{}'.format(i) for i in os.listdir(train_dir) if 'dog' in i]  # get dog images
train_cats = ['/Users/denis/PycharmProjects/diplom/input/train/{}'.format(i) for i in os.listdir(train_dir) if 'cat' in i]  # get cat images

test_imgs = ['/Users/denis/PycharmProjects/diplom/input/test/{}'.format(i) for i in os.listdir(test_dir)]  # get test images

print("dogs total {}, cats total {}".format(len(train_dogs), len(train_cats)))
train_imgs = train_dogs[:500] + train_cats[:500]  # slice the dataset and use 2000 in each class
random.shuffle(train_imgs)  # shuffle it randomly

# Clear list that are useless
del train_dogs
del train_cats
gc.collect()  # collect garbage to save memory

# Lets declare our image dimensions
# we are using coloured images.
nrows = 150
ncolumns = 150
channels = 3  # change to 1 if you want to use grayscale image


# A function to read and process the images to an acceptable format for our model
def read_and_process_image(list_of_images, rows, columns):
    """
    Returns two arrays:
        X is an array of resized images
        y is an array of labels
    """
    X = []  # images
    y = []  # labels
    bgr_images = []
    for image in list_of_images:
        # print("l")
        img = (cv2.resize(cv2.imread(image, cv2.IMREAD_COLOR), (rows, columns),
                            interpolation=cv2.INTER_CUBIC))  # Read the image
        bgr_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        X.append(img)
        bgr_images.append(bgr_img)
        # get the labels
        if 'dog' in image:
            y.append(1)
        elif 'cat' in image:
            y.append(0)

    return X, y, bgr_images


# get the train and label data
X, y, _ = read_and_process_image(train_imgs,nrows, ncolumns)

# Lets view some of the pics
# plt.figure(figsize=(20, 10))
# columns = 5
# for i in range(columns):
#     plt.subplot(int(5 / columns + 1), columns, i + 1)
#     plt.imshow(X[i])

del train_imgs
gc.collect()

# Convert list to numpy array
X = np.array(X)
y = np.array(y)

# Lets plot the label to be sure we just have two class
print("Shape of train images is:", X.shape)
print("Shape of labels is:", y.shape)

# Lets split the data into train and test set
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=2)

print("Shape of train images is:", X_train.shape)
print("Shape of validation images is:", X_val.shape)
print("Shape of labels is:", y_train.shape)
print("Shape of labels is:", y_val.shape)

# clear memory
del X
del y
gc.collect()

# get the length of the train and validation data
ntrain = len(X_train)
nval = len(X_val)

# We will use a batch size of 32. Note: batch size should be a factor of 2.***4,8,16,32,64...***
batch_size = 32

# def fitness(payload, X_train, X_val, y_train, y_val, epochs=16, nrows=150, ncolumns=150, batch_size=32):
#     import tensorflow
#     from keras import layers
#     from keras import models
#     from keras import optimizers
#     from keras.preprocessing.image import ImageDataGenerator
#     from keras.preprocessing.image import img_to_array, load_img
#
#     ntrain = len(X_train)
#     nval = len(X_val)
#     color_layers = 3
#
#     model = models.Sequential()
#     model.add(layers.Reshape((nrows, ncolumns, color_layers), input_shape=(nrows, ncolumns, color_layers)))
#
#     # тут были слои, а будет - нагрузка от внешних аргументов
#     layers_kk = payload[0]
#     optimizer = payload[1]
#     print("payload reading")
#     print(layers_kk)
#     print(payload)
#     for layer in layers_kk:
#         print("layer type", type(layer))
#         print(layer)
#         # exec(layer)
#     # exec(optimizer)
#     model.add(layers.Conv2D(32, (3, 3), activation='relu'))
#     model.add(layers.MaxPooling2D((2, 2)))
#     model.add(layers.Conv2D(64, (3, 3), activation='relu'))
#     model.add(layers.MaxPooling2D((2, 2)))
#     model.add(layers.Conv2D(128, (3, 3), activation='relu'))
#     model.add(layers.MaxPooling2D((2, 2)))
#     model.add(layers.Flatten())
#     model.add(layers.Dropout(0.5))  # Dropout for regularization
#     initial = tensorflow.keras.initializers.RandomNormal(mean=0.5, stddev=0)
#     model.add(layers.Dense(512, activation='relu', kernel_initializer=initial))
#     # model.add(layers.Dropout(0.5))
#     model.add(layers.Dense(1, activation='sigmoid'))  # Sigmoid function at the end because we have just two classes
#     model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
#     # model.summary()
#
#     # model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
#     # Lets create the augmentation configuration
#     # This helps prevent overfitting, since we are using a small dataset
#     train_datagen = ImageDataGenerator(rescale=1. / 255,  # Scale the image between 0 and 1
#                                        rotation_range=40,
#                                        width_shift_range=0.2,
#                                        height_shift_range=0.2,
#                                        shear_range=0.2,
#                                        zoom_range=0.2,
#                                        horizontal_flip=True, )
#
#     val_datagen = ImageDataGenerator(rescale=1. / 255)  # We do not augment validation data. we only perform rescale
#
#     # Create the image generators
#     train_generator = train_datagen.flow(X_train, y_train, batch_size=batch_size)
#     val_generator = val_datagen.flow(X_val, y_val, batch_size=batch_size)
#
#     # The training part
#     # We train for 64 epochs with about 100 steps per epoch
#     history = model.fit_generator(train_generator,
#                                   steps_per_epoch=ntrain // batch_size,
#                                   epochs=1,
#                                   validation_data=val_generator,
#                                   validation_steps=nval // batch_size)
#
#     # acc = history.history['acc']
#     val_acc = history.history['val_acc']
#     # loss = history.history['loss']
#     # val_loss = history.history['val_loss']
#     return val_acc
accuracy_list = []
for i in range(0, 50):
    specie = genetic_algorithm.create_specie()
    payload = genetic_algorithm.convert_layer_list(specie[0], specie[1])
    accuracy = neural_fitness.fitness(payload=payload, X_train=X_train, X_val=X_val, y_train=y_train, y_val=y_val, epochs=1)
    specie.clear()
    payload.clear()
    accuracy_list.append(accuracy[-1])
print(accuracy_list)
print(max(accuracy_list))
