import random

optimizers = [
    'Adadelta',
    'Adagrad',
    'Adam',
    'Adamax',
    'Ftrl',
    'Nadam',
    'RMSProp', #fine
    'SGD',
]

losses = [
    'binary_crossentropy', #fine
    'binary_focal_crossentropy',
    'categorical_crossentropy',
    'categorical_hinge',
    'cosine_similarity',
    'hinge',
    'huber',
    'KLDivergence',
    'LogCosh',
    'MeanAbsoluteError',
    'MeanAbsolutePercentageError',
    'MeanSquaredError',
    'MeanSquaredLogarithmicError',
    'Poisson',
    'SparseCategoricalCrossentropy',
    'SquaredHinge'
]



convolutional_layer = {
        'convolution': 'Conv2D(',
        'size': 0,
        'activation': 'relu',
        'pooling': 'MaxPooling'
}

# print(convolutional_layer['convolution'])
activations = [
    'relu',
    'sigmoid',
    'softmax',
    'softplus',
    'softsign',  # чет странное
    'tanh',
    'selu',
    'elu',
    'exponential',
]

poolings = [
    'MaxPooling2D((2, 2)',  # вроде работает
    'AveragePooling2D((2, 2)',  # вроде работает
    # 'GlobalAveragePooling2D((2, 2)', # не работает
    # 'GlobalMaxPooling2D((2, 2)' # не работает тоже
]




def create_random_conv_layer():
    layer = {
        'tag': 'conv_layer',
        'convolution': 'Conv2D(',
        'size': 2 ** random.randint(4, 7),
        'activation': activations[random.randint(0, len(activations)-1)],
        'pooling': poolings[random.randint(0, len(poolings)-1)]
    }
    return layer


def output_conv_layer(layer):
    # model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    # model.add(layers.MaxPooling2D((2, 2)))
    out = "model.add(layers."+layer['convolution']+str(layer['size'])+", (3, 3), activation='"+layer['activation']+"')) \nmodel.add(layers."+layer['pooling']+"))"
    out2 = "model.add(layers."+layer['pooling']+"))"
    # print(out)
    # print(out2)
    # out = out + "\ " + out2
    # print(out)
    return out


def create_flatten_layer():
    layer = {
        'tag': 'flatten_layer',
        'payload': 'Flatten()'
    }
    return layer


def output_flatten_layer(layer):
    if layer['tag'] != 'flatten_layer':
        print("wrong tag. Expected 'flatten_layer', got {}".format(layer['tag']))
        raise Exception
    else:
        out = "model.add(layers." + layer['payload'] + ")"
        # print(out)
        return out

def create_dense_layer():
    # initial = tensorflow.keras.initializers.RandomNormal(mean=0.5, stddev=0)
    # model.add(layers.Dense(512, activation='relu', kernel_initializer=initial))
    layer = {
        'tag': 'dense_layer',
        'payload': 'Dense(',
        'parameters': 2 ** random.randint(3, 12),
        'activation': activations[random.randint(0, len(activations)-1)]
    }
    return layer


def output_dense_layer(layer):
    if layer['tag'] != 'dense_layer':
        print("wrong tag. Expected 'dense_layer', got {}".format(layer['tag']))
        raise Exception
    else:
        out = "model.add(layers." + layer['payload']+str(layer['parameters']) + ", activation='"+ layer['activation'] + "'))"
        # print(out)
        return out

def create_dropout_layer():
    layer = {
        'tag': 'dropout_layer',
        'payload': 'Dropout(',
        'value': random.randint(0, 800)/1000
    }
    return layer


def output_dropout_layer(layer):
    if layer['tag'] != 'dropout_layer':
        print("wrong tag. Expected 'dropout_layer', got {}".format(layer['tag']))
        raise Exception
    else:
        out = "model.add(layers." + layer['payload']+str(layer['value']) + "))"
        # print(out)
        return out


def create_output_layer():
    layer = {
        'tag': 'output_layer',
        'payload': "model.add(layers.Dense(1, activation='sigmoid'))"
    }
    return layer


def output_output_layer(layer):
    if layer['tag'] != 'output_layer':
        print("wrong tag. Expected 'output_layer', got {}".format(layer['tag']))
        raise Exception
    else:
        return layer['payload']


def create_optimizer():
    optimizer = {
        'tag': 'optimizer',
        'loss': losses[random.randint(0, len(losses)-1)],
        'optimizer': optimizers[random.randint(0, len(optimizers)-1)]
    }
    return optimizer

def output_optimizer(optimizer):
    if optimizer['tag'] != 'optimizer':
        print("wrong tag. Expected 'optimizer', got {}".format(optimizer['tag']))
        raise Exception
    else:
        return "model.compile(loss='{}', optimizer='{}', metrics=['acc'])".format(optimizer['loss'], optimizer['optimizer'])


print(output_optimizer(create_optimizer()))
def create_specie():
    # model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
    layer_list = []
    layer_list.append(create_random_conv_layer())
    layer_list.append(create_random_conv_layer())
    layer_list.append(create_random_conv_layer())
    layer_list.append(create_flatten_layer())
    layer_list.append(create_dropout_layer())
    layer_list.append(create_dense_layer())
    layer_list.append(create_output_layer())

    compilator = create_optimizer()
    # print(layer_list)

    return [layer_list, compilator]


def convert_layer_list(layer_list, compilator):
    out = []
    comp = output_optimizer(compilator)
    for layer in layer_list:
        if layer['tag'] == 'conv_layer':
            out.append(output_conv_layer(layer))
        elif layer['tag'] == 'flatten_layer':
            out.append(output_flatten_layer(layer))
        elif layer['tag'] == 'dense_layer':
            out.append(output_dense_layer(layer))
        elif layer['tag'] == 'dropout_layer':
            out.append(output_dropout_layer(layer))
        elif layer['tag'] == 'output_layer':
            out.append(output_output_layer(layer))
        else:
            raise Exception
    return [out, comp]


specie = create_specie()
convert_layer_list(specie[0], specie[1])

