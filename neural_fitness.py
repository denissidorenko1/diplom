
def fitness(payload, X_train, X_val, y_train, y_val, epochs=16, nrows=150, ncolumns=150, batch_size=32):
    from keras import layers
    from keras import models
    from keras import optimizers
    from keras.preprocessing.image import ImageDataGenerator
    from keras.preprocessing.image import img_to_array, load_img

    ntrain = len(X_train)
    nval = len(X_val)
    color_layers = 3

    model = models.Sequential()
    model.add(layers.Reshape((nrows, ncolumns, color_layers), input_shape=(nrows, ncolumns, color_layers)))

    # тут были слои, а будет - нагрузка от внешних аргументов
    payloaded_layers = payload[0]
    payloaded_optimizer = payload[1]
    print("payload reading")

    for layer in payloaded_layers:
        print(layer)
        exec(layer)
    print(payloaded_optimizer)
    exec(payloaded_optimizer)
    # Lets see our model
    # model.summary()

    # We'll use the RMSprop optimizer with a learning rate of 0.0001
    # We'll use binary_crossentropy loss because its a binary classification
    # model.compile(loss='binary_crossentropy', optimizer=optimizers.RMSprop(lr=1e-4), metrics=['acc'])
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
    # model.summary()

    # model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
    # Lets create the augmentation configuration
    # This helps prevent overfitting, since we are using a small dataset
    train_datagen = ImageDataGenerator(rescale=1. / 255,  # Scale the image between 0 and 1
                                       rotation_range=40,
                                       width_shift_range=0.2,
                                       height_shift_range=0.2,
                                       shear_range=0.2,
                                       zoom_range=0.2,
                                       horizontal_flip=True, )

    val_datagen = ImageDataGenerator(rescale=1. / 255)  # We do not augment validation data. we only perform rescale

    # Create the image generators
    train_generator = train_datagen.flow(X_train, y_train, batch_size=batch_size)
    val_generator = val_datagen.flow(X_val, y_val, batch_size=batch_size)

    # The training part
    # We train for 64 epochs with about 100 steps per epoch
    history = model.fit_generator(train_generator,
                                  steps_per_epoch=ntrain // batch_size,
                                  epochs=epochs,
                                  validation_data=val_generator,
                                  validation_steps=nval // batch_size)

    # acc = history.history['acc']
    val_acc = history.history['val_acc']
    # loss = history.history['loss']
    # val_loss = history.history['val_loss']
    return val_acc


