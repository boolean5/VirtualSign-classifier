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


def first_col_to_last(dataframe, columns):
    # Swap id column to the last place TODO: Have a standard way of doing this, probably according to Tiago's VS script
    cols = dataframe.columns.tolist()
    cols = cols[1:] + cols[:1]
    df = dataframe[cols]
    df.columns = columns
    return df


def create_dataset(path, deletedups=False, randomize=True, drop_digits=6, raw=False):
    import pandas as pd
    import os

    sensors = ['id', 'thu-near', 'thu-far', 'thu-ind', 'ind-near', 'ind-far', 'ind-mid', 'mid-near', 'mid-far',
               'mid-rin', 'rin-near', 'rin-far', 'rin-lil', 'lil-near', 'lil-far']
    frames = []
    if os.path.isdir(path):
        for root, dirs, files in os.walk(path):
            for filename in files:
                with open(os.path.join(root, filename)) as infile:
                    df = pd.read_csv(infile, delim_whitespace=True, names=sensors, usecols=sensors, header=None)
                frames.append(df)
    else:
        with open(path) as infile:
            df = pd.read_csv(infile, delim_whitespace=True, names=sensors, usecols=sensors, header=None)
            frames.append(df)

    dataset = pd.concat(frames, ignore_index=True)

    if raw:
        dataset.iloc[:, 1:] = dataset.iloc[:, 1:].divide(4096)  # 4096 is the glove max output value reported by 5DT

    if deletedups:
        dataset = dataset.drop_duplicates().reset_index(drop=True)

    # TODO: This randomization has to be stratified
    if randomize:
        dataset = dataset.sample(frac=1).reset_index(drop=True)

    if drop_digits:
        dataset = dataset.round(drop_digits)

    return dataset


def build_model(model_type, input_dim, output_dim):
    if model_type == 'inception':
        model = build_inception_layer(input_dim, output_dim)
    elif model_type == 'seq_v1':
        model = build_sequential_v1(input_dim, output_dim)
    elif model_type == 'seq_v2':
        model = build_sequential_v2(input_dim, output_dim)
    elif model_type == 'functional':
        model = build_functional(input_dim, output_dim)
    else:
        raise Exception('Expected one of [inception, seq_v1, seq_v2, functional] model type literals')

    return model


def build_sequential_v1(input_dim, output_dim):
    from keras.models import Sequential
    from keras.layers import Conv1D, MaxPool1D, Flatten, Dense
    from keras.activations import relu, softmax

    filter_size_1 = 2
    filter_size_2 = 5
    output_size_1 = 20
    output_size_2 = 10
    pool_size_1 = 2
    feature_map_size = 200

    model = Sequential()

    model.add(Conv1D(output_size_1, filter_size_1, input_shape=(input_dim, 1), activation=relu))
    model.add(MaxPool1D(pool_size=pool_size_1, strides=1))

    model.add(Conv1D(output_size_2, filter_size_2, activation=relu))

    model.add(Flatten())
    model.add(Dense(feature_map_size, activation=relu))
    model.add(Dense(output_dim, activation=softmax))

    return model


def build_sequential_v2(input_dim, output_dim):
    from keras.models import Sequential
    from keras.layers import Conv1D, Flatten, Dense, BatchNormalization, Dropout
    from keras.activations import softmax, selu

    ACTIVATION_FUNCTION = selu
    output_size = 32
    feature_map_size = 200

    model = Sequential()
    model.add(BatchNormalization(input_shape=(input_dim, 1)))
    model.add(Conv1D(output_size, 1, input_shape=(input_dim, 1), activation=ACTIVATION_FUNCTION))
    model.add(Dropout(0.3))
    model.add(Conv1D(output_size, 3, input_shape=(input_dim, 1), activation=ACTIVATION_FUNCTION))
    model.add(Dropout(0.3))
    model.add(Conv1D(output_size, 3, input_shape=(input_dim, 1), activation=ACTIVATION_FUNCTION))
    model.add(Dropout(0.3))

    model.add(Flatten())
    model.add(Dense(feature_map_size, activation=ACTIVATION_FUNCTION))
    model.add(Dropout(0.3))
    model.add(Dense(output_dim, activation=softmax))

    return model


def build_functional(input_dim, output_dim):
    from keras.models import Model
    from keras.regularizers import l1_l2
    from keras.layers import Conv1D, Flatten, Dropout, Dense, Input, concatenate, BatchNormalization
    from keras.activations import softmax, selu

    DROPOUT_RATE = 0.5
    ACTIVATION_FUNCTION = selu
    output_size = 16
    feature_map_size = 100

    inputs = Input(shape=(input_dim, 1))

    bn_input = BatchNormalization()(inputs)

    conv_1 = Conv1D(output_size, 1, activation=ACTIVATION_FUNCTION, kernel_regularizer=l1_l2(0.01))(bn_input)
    conv_1 = Dropout(DROPOUT_RATE)(conv_1)

    conv_2 = Conv1D(output_size, 1, activation=ACTIVATION_FUNCTION, kernel_regularizer=l1_l2(0.01))(bn_input)
    conv_2 = Conv1D(output_size, 3, activation=ACTIVATION_FUNCTION, kernel_regularizer=l1_l2(0.01))(conv_2)
    conv_2 = Conv1D(output_size, 3, activation=ACTIVATION_FUNCTION, kernel_regularizer=l1_l2(0.01))(conv_2)
    conv_2 = Dropout(DROPOUT_RATE)(conv_2)

    output = concatenate([conv_1, conv_2], axis=1)

    flatten = Flatten()(output)
    dense = Dense(feature_map_size, activation=ACTIVATION_FUNCTION, kernel_regularizer=l1_l2(0.01))(flatten)
    dense = Dropout(DROPOUT_RATE)(dense)
    output = Dense(output_dim, activation=softmax)(dense)

    model = Model(inputs=inputs, outputs=output)
    return model


def build_inception_layer(input_dim, output_dim):
    from keras.models import Model
    from keras.regularizers import l1_l2
    from keras.layers import Conv1D, MaxPool1D, Flatten, Dropout, Dense, Input, concatenate, BatchNormalization
    from keras.activations import softmax, elu

    # TODO: Consider 'same', 'valid' options for padding

    DROPOUT_RATE = 0.3
    ACTIVATION_FUNCTION = elu
    output_size = 16
    pool_size = 3
    feature_map_size = 50

    inputs = Input(shape=(input_dim, 1))

    bn_input = BatchNormalization()(inputs)

    conv_1 = Conv1D(output_size, 1, activation=ACTIVATION_FUNCTION, kernel_regularizer=l1_l2(0.01))(bn_input)
    conv_1 = Dropout(DROPOUT_RATE)(conv_1)

    conv_2 = Conv1D(output_size, 1, activation=ACTIVATION_FUNCTION, kernel_regularizer=l1_l2(0.01))(bn_input)
    conv_2 = Conv1D(output_size, 3, activation=ACTIVATION_FUNCTION, kernel_regularizer=l1_l2(0.01))(conv_2)
    conv_2 = Dropout(DROPOUT_RATE)(conv_2)

    conv_3 = Conv1D(output_size, 1, activation=ACTIVATION_FUNCTION, kernel_regularizer=l1_l2(0.01))(bn_input)
    conv_3 = Conv1D(output_size, 3, activation=ACTIVATION_FUNCTION, kernel_regularizer=l1_l2(0.01))(conv_3)
    conv_3 = Conv1D(output_size, 3, activation=ACTIVATION_FUNCTION, kernel_regularizer=l1_l2(0.01))(conv_3)
    conv_3 = Dropout(DROPOUT_RATE)(conv_3)

    pool_1 = MaxPool1D(pool_size, strides=1, padding='same')(bn_input)
    conv_4 = Conv1D(output_size, 1, activation=ACTIVATION_FUNCTION, kernel_regularizer=l1_l2(0.01))(pool_1)
    conv_4 = Dropout(DROPOUT_RATE)(conv_4)

    output = concatenate([conv_1, conv_2, conv_3, conv_4], axis=1)

    flatten = Flatten()(output)
    dense = Dense(feature_map_size, activation=ACTIVATION_FUNCTION, kernel_regularizer=l1_l2(0.01))(flatten)
    dense = Dropout(DROPOUT_RATE)(dense)
    output = Dense(output_dim, activation=softmax)(dense)

    model = Model(inputs=inputs, outputs=output)
    return model


def last_to_first():
    df = create_dataset('datasets/old_datasets/dataset_marcelo_RG.txt', deletedups=False, randomize=False)
    df.columns = ['thu-near', 'thu-far', 'thu-ind', 'ind-near', 'ind-far', 'ind-mid', 'mid-near', 'mid-far',
                  'mid-rin', 'rin-near', 'rin-far', 'rin-lil', 'lil-near', 'lil-far', 'id']

    cols = df.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    df = df[cols]

    df.to_csv('marcelo-r-scaled.txt', sep='\t', header=False, index=False, float_format='%.3f')
