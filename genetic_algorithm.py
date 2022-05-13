import gc
import random

optimizers = [
    'Adadelta',
    'Adagrad',
    'Adam',
    # 'Adamax', # crashed
    # 'Ftrl', # crashed
    'Nadam',
    'RMSProp', #fine
    'SGD',
]

losses = [
    'binary_crossentropy', #fine
    'binary_focal_crossentropy',
    'categorical_crossentropy',
    'categorical_hinge',
    # 'cosine_similarity', # crash
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


def create_specie(test_mode=False):
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

    # return [layer_list, compilator]
    if test_mode:
        inst = {
            'layers': layer_list,
            'optimizer': compilator,
            'val_acc': random.random()
        }
        return inst

    inst = {
        'layers': layer_list,
        'optimizer': compilator,
        'val_acc': 0
    }
    return inst

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


def create_population(population_size=40):
    population = []
    for i in range(0, population_size):
        population.append(create_specie(test_mode=True))
    return population


def select_half_best(population):
    """Оператор отбора. Отбирает половину лучших особей"""
    # sorted(list_to_be_sorted, key=lambda d: d['name'])
    pop_size = len(population)
    best_species = []
    sorted_population = sorted(population, key=lambda d: d['val_acc'])
    # for specie in sorted_population:
        # print("val_acc: ", specie['val_acc'])
    # print("lllll")
    for i in range(0, int(pop_size/2)):
        best_species.append(sorted_population.pop())
    # for specie in best_species:
        # print("val_acc: ", specie['val_acc'])
    del population
    gc.collect()
    return best_species


def breed(male, female):
    """
    Получили двух особей. У них под атрибутом layer имеется массив словарей, содержащих слои и компилятор
    Наш способ - придумать двух отпрысков: у которых слои будут случайно браться от одного из родителей
    """  # for i in range(0, len(male_layers)):
    #     if random.random() > 0.5:
    #
    #         kid_1_layers.append(male_layers[i])
    #     else:
    #         kid_1_layers.append(female_layers[i])
    #     if random.random() > 0.5:
    #         kid_2_layers.append(male_layers[i])
    #     else:
    #         kid_2_layers.append(female_layers[i])

    # male_first_conv_layer = male_layers[0]
    # conv_layer_example = {
    #     'tag': 'conv_layer',
    #     'convolution': 'Conv2D(',
    #     'size': 2 ** random.randint(4, 7),
    #     'activation': activations[random.randint(0, len(activations) - 1)],
    #     'pooling': poolings[random.randint(0, len(poolings) - 1)]
    # }
    male_layers = male['layers']
    male_optimizer = male['optimizer']
    female_layers = female['layers']
    female_optimizer = female['optimizer']

    kid_1_layers = []
    kid_2_layers = []



    # Собираем сверточные слои:
    for i in range(0, 3):
        temp_male_conv_layer = male_layers[i]
        temp_female_conv_layer = female_layers[i]
        if random.random() > 0.5:
            temp_size = temp_male_conv_layer['size']
        else:
            temp_size = temp_female_conv_layer['size']
        if random.random() > 0.5:
            temp_activation = temp_male_conv_layer['activation']
        else:
            temp_activation = temp_female_conv_layer['activation']
        if random.random() > 0.5:
            temp_pooling = temp_male_conv_layer['pooling']
        else:
            temp_pooling = temp_female_conv_layer['pooling']
        conv_layer_temp = {
            'tag': 'conv_layer',
            'convolution': 'Conv2D(',
            'size': temp_size,
            'activation': temp_activation,
            'pooling': temp_pooling
        }
        kid_1_layers.append(conv_layer_temp)
    kid_1_layers.append(male_layers[3])  # flatten layer is stable in all species

    if random.random() > 0.5:  # dropout layer has only one variable
        kid_1_layers.append(male_layers[4])
    else:
        kid_1_layers.append(female_layers[4])


    # Плотный слой имеет 2 параметра
    temp_male_dense_layer = male_layers[5]
    temp_female_dense_layer = female_layers[5]

    if random.random() > 0.5:
        temp_parameters = temp_male_dense_layer['parameters']
    else:
        temp_parameters = temp_female_dense_layer['parameters']
    if random.random() > 0.5:
        temp_dense_activation = temp_male_dense_layer['activation']
    else:
        temp_dense_activation = temp_female_dense_layer['activation']

    kid_1_temp_dense_layer = {
        'tag': 'dense_layer',
        'payload': 'Dense(',
        'parameters': temp_parameters,
        'activation': temp_dense_activation
    }
    kid_1_layers.append(kid_1_temp_dense_layer)
    kid_1_layers.append(male_layers[6])





    # Собираем сверточные слои:
    for i in range(0, 3):
        temp_male_conv_layer = male_layers[i]
        temp_female_conv_layer = female_layers[i]
        if random.random() > 0.5:
            temp_size = temp_male_conv_layer['size']
        else:
            temp_size = temp_female_conv_layer['size']
        if random.random() > 0.5:
            temp_activation = temp_male_conv_layer['activation']
        else:
            temp_activation = temp_female_conv_layer['activation']
        if random.random() > 0.5:
            temp_pooling = temp_male_conv_layer['pooling']
        else:
            temp_pooling = temp_female_conv_layer['pooling']
        conv_layer_temp = {
            'tag': 'conv_layer',
            'convolution': 'Conv2D(',
            'size': temp_size,
            'activation': temp_activation,
            'pooling': temp_pooling
        }
        kid_2_layers.append(conv_layer_temp)
    kid_2_layers.append(male_layers[3])  # flatten layer is stable in all species

    if random.random() > 0.5:  # dropout layer has only one variable
        kid_2_layers.append(male_layers[4])
    else:
        kid_2_layers.append(female_layers[4])

    # Плотный слой имеет 2 параметра
    temp_male_dense_layer = male_layers[5]
    temp_female_dense_layer = female_layers[5]

    if random.random() > 0.5:
        temp_parameters = temp_male_dense_layer['parameters']
    else:
        temp_parameters = temp_female_dense_layer['parameters']
    if random.random() > 0.5:
        temp_dense_activation = temp_male_dense_layer['activation']
    else:
        temp_dense_activation = temp_female_dense_layer['activation']

    kid_2_temp_dense_layer = {
        'tag': 'dense_layer',
        'payload': 'Dense(',
        'parameters': temp_parameters,
        'activation': temp_dense_activation
    }
    kid_2_layers.append(kid_2_temp_dense_layer)
    kid_2_layers.append(male_layers[6])

    if random.random() > 0.5:
        kid_1_loss = male_optimizer['loss']
    else:
        kid_1_loss = female_optimizer['loss']
    if random.random() > 0.5:
        kid_2_loss = male_optimizer['loss']
    else:
        kid_2_loss = female_optimizer['loss']

    if random.random() > 0.5:
        kid_1_opt = male_optimizer['optimizer']
    else:
        kid_1_opt = female_optimizer['optimizer']
    if random.random() > 0.5:
        kid_2_opt = male_optimizer['optimizer']
    else:
        kid_2_opt = female_optimizer['optimizer']

    kid_1_optimizer = {
        'tag': 'optimizer',
        'loss': kid_1_loss,
        'optimizer': kid_1_opt
    }
    kid_2_optimizer = {
        'tag': 'optimizer',
        'loss': kid_2_loss,
        'optimizer': kid_2_opt
    }
    first = {
        'layers': kid_1_layers,
        'optimizer': kid_1_optimizer,
        'val_acc': 0
    }
    second = {
        'layers': kid_2_layers,
        'optimizer': kid_2_optimizer,
        'val_acc': 0
    }
    return first, second


def crossover(best_species):
    offsprings = []
    counter = len(best_species)
    # print()
    while counter != 0:
        first, second = breed(best_species[random.randint(0, len(best_species)-1)],
                              best_species[random.randint(0, len(best_species)-1)])
        offsprings.append(first)
        offsprings.append(second)
        counter -= 1
    # print("best species count:", len(best_species))
    # print("offsprings count:", len(offsprings))
    return offsprings


pop = create_population(20)
kekeke = select_half_best(pop)
# dick, cock =breed(kekeke[0], kekeke[1])
# print("*"*60)
# print(dick)
# print()
# print(cock)
crossover(kekeke)
# specie = create_specie()
# convert_layer_list(specie['layers'], specie['optimizer']) # this is outdated version which uses list instead of dict

