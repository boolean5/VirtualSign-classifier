#!/usr/bin/env python
# -*- coding: utf-8 -*-

def correct_dict():
    replacements = {'Ã£': 'ã', 'Ã¡': 'á', 'Ã¢': 'â', 'Ã§': 'ç', 'Ãª': 'ê', 'Ã©': 'é',
                    'Ã³': 'ó', 'Ã­': 'í', 'Ãº': 'ú', 'Ã': 'Á', 'Ãµ': 'õ'}

    with open('data/datasetLeft.txt') as infile, open('data/dictLeft-corrected', 'w') as outfile:
        for line in infile:
            for src, target in replacements.items():
                line = line.replace(src, target)
            outfile.write(line)


def duplicates(dataframe):
    import numpy as np

    num_dups = np.flatnonzero(dataframe.duplicated())
    return len(num_dups)


def create_dataset(path, deletedups=True, randomize=True):
    import pandas as pd
    import os

    sensors = ['col'+str(i) for i in range(14)] + ['id']
    frames = []

    for root, dirs, files in os.walk(path):
        for filename in files:
            with open(os.path.join(root, filename)) as infile:
                df = pd.read_csv(infile, delim_whitespace=True, names=sensors, usecols=sensors, header=None)

            # Check for misplaced class-label column, properly swap columns/shift contents if so
            if df.iloc[:, 0].dtype == 'int64':

                # Reorder columns, then change back to original names
                cols = df.columns.tolist()
                cols = cols[1:] + cols[:1]
                df = df[cols]
                df.columns = sensors
            frames.append(df)

    dataset = pd.concat(frames, ignore_index=True)

    if deletedups:
        dataset = dataset.drop_duplicates().reset_index(drop=True)

    if randomize:
        dataset = dataset.sample(frac=1).reset_index(drop=True)

    return dataset


def build_sequential_v1(input_dim, output_dim):
    from keras.models import Sequential
    from keras.layers import Conv1D, MaxPool1D, AveragePooling1D, Flatten, Dropout, Dense
    from keras.activations import relu, softmax

    filter_size_1 = 2
    filter_size_2 = 5
    output_size_1 = 20
    output_size_2 = 10
    pool_size_1 = 2
    pool_size_2 = 1
    feature_map_size = 200

    model = Sequential()

    model.add(Conv1D(output_size_1, filter_size_1, input_shape=(input_dim, 1), activation=relu))
    model.add(MaxPool1D(pool_size=pool_size_1, strides=1))
    # model.add(Dropout(0.5))

    model.add(Conv1D(output_size_2, filter_size_2, activation=relu))
    model.add(MaxPool1D(pool_size=pool_size_2, strides=1))
    # model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(feature_map_size, activation=relu))
    model.add(Dense(output_dim, activation=softmax))

    return model


def build_sequential_v2():
    pass


def build_inception_like(input_dim, output_dim):
    from keras.models import Model
    from keras.layers import Conv1D, MaxPool1D, AveragePooling1D, Flatten, Dropout, Dense, Input, concatenate
    from keras.activations import relu, softmax

    # TODO: Create a for loop for multiple filter implementation
    # TODO: Consider 'same', 'valid' options for padding

    DROPOUT_RATE = 0.5
    ACTIVATION_FUNCTION = relu
    output_size = 32
    pool_size = 3
    feature_map_size = 200

    inputs = Input(shape=(input_dim, 1))

    conv_1 = Conv1D(output_size, 1, activation=ACTIVATION_FUNCTION)(inputs)
    conv_1 = Dropout(DROPOUT_RATE)(conv_1)

    conv_2 = Conv1D(output_size, 1, activation=ACTIVATION_FUNCTION)(inputs)
    conv_2 = Conv1D(output_size, 3, activation=ACTIVATION_FUNCTION)(conv_2)
    conv_2 = Dropout(DROPOUT_RATE)(conv_2)

    conv_3 = Conv1D(output_size, 1, activation=ACTIVATION_FUNCTION)(inputs)
    conv_3 = Conv1D(output_size, 5, activation=ACTIVATION_FUNCTION)(conv_3)
    conv_3 = Dropout(DROPOUT_RATE)(conv_3)

    pool_1 = MaxPool1D(pool_size, strides=1, padding='same')(inputs)
    conv_4 = Conv1D(output_size, 1, activation=relu)(pool_1)
    conv_4 = Dropout(DROPOUT_RATE)(conv_4)

    output = concatenate([conv_1, conv_2, conv_3, conv_4], axis=1)

    flatten = Flatten()(output)
    dense = Dense(feature_map_size, activation=relu)(flatten)
    dense = Dropout(DROPOUT_RATE)(dense)
    output = Dense(output_dim, activation=softmax)(dense)

    model = Model(inputs=inputs, output=output)
    return model
