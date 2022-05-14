import gc
import time

from keras import layers
from keras import models
# from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
# from keras.preprocessing.image import img_to_array, load_img

def fitness(payload, X_train, X_val, y_train, y_val, epochs=16, nrows=150, ncolumns=150, batch_size=32):
    gc.set_threshold(400, 15, 15)
    ntrain = len(X_train)
    nval = len(X_val)
    color_layers = 3

    model = models.Sequential()
    model.add(layers.Reshape((nrows, ncolumns, color_layers), input_shape=(nrows, ncolumns, color_layers)))

    # тут были слои, а будет - нагрузка от внешних аргументов
    payloaded_layers = payload[0]
    payloaded_optimizer = payload[1]
    print("payload reading")

    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dropout(0.5))  # Dropout for regularization
    model.add(layers.Dense(512, activation='relu'))
    # model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1, activation='sigmoid'))  # Sigmoid function at the end because we have just two classes
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])


    for layer in payloaded_layers:
        pass
        # print(layer)
        # exec(layer)
    # print(payloaded_optimizer)
    # exec(payloaded_optimizer)
    model.summary()
    # Lets create the augmentation configuration
    # This helps prevent overfitting, since we are using a small dataset
    # Scale the image between 0 and 1
    train_datagen = ImageDataGenerator(rescale=1. / 255, rotation_range=40,width_shift_range=0.2, height_shift_range=0.2,
                                       shear_range=0.2, zoom_range=0.2, horizontal_flip=True, )
    #
    val_datagen = ImageDataGenerator(rescale=1. / 255)  # We do not augment validation data. we only perform rescale


    # Create the image generators
    train_generator = train_datagen.flow(X_train, y_train, batch_size=batch_size)
    val_generator = val_datagen.flow(X_val, y_val, batch_size=batch_size)

    # The training part
    # We train for 64 epochs with about 100 steps per epoch
    # history = model.fit(x=X_train, y=y_train, validation_split=0.2, epochs=epochs, steps_per_epoch=ntrain // batch_size,
    #                     validation_steps=nval // batch_size)

    history = model.fit_generator(train_generator, steps_per_epoch=ntrain // batch_size, epochs=epochs, validation_data=val_generator, validation_steps=nval // batch_size)

    print(gc.get_count())
    del model
    del val_datagen
    del train_datagen
    del train_generator
    del val_generator
    del payloaded_layers
    del payloaded_optimizer
    del payload
    print(gc.get_count())
    gc.collect()
    print(gc.get_count())
    time.sleep(10)
    return history.history['val_acc'][-1]

