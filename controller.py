import time

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
train_imgs = train_dogs[:2000] + train_cats[:2000]  # slice the dataset and use 2000 in each class
# random.shuffle(train_imgs)  # shuffle it randomly

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
average_accuracy_list = []
# accuracy_list = []
gc.set_threshold(400, 15, 15)
try:

    population = genetic_algorithm.create_population(8)
    # time.sleep(100)
    for i in range(0, 20):
        average_cum = 0
        for specie in population:
            payload = genetic_algorithm.convert_layer_list(specie['layers'], specie['optimizer'])
            val_accuracy = neural_fitness.fitness(payload=payload, X_train=X_train, X_val=X_val, y_train=y_train,
                                                  y_val=y_val, epochs=2)

            # payload.clear()
            del payload
            # time.sleep(2)
            specie['val_acc'] = val_accuracy
            average_cum += val_accuracy
            gc.collect()
        average_cum /= len(population)
        average_accuracy_list.append(average_cum)
        print("population_len: {}".format(len(population)))
        print("population num: {}, average_accuracy={}".format(i, average_cum))
        population = genetic_algorithm.crossover(genetic_algorithm.select_half_best(population))
        gc.collect()
            # accuracy_list.append(val_accuracy)
        # print("specie val acc:", specie['val_acc'])
except KeyboardInterrupt:
    pass
print(average_accuracy_list)
# print(accuracy_list)
# print(max(accuracy_list))
