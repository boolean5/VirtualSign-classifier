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
    cols = dataframe.columns.tolist()
    cols = cols[1:] + cols[:1]
    df = dataframe[cols]
    df.columns = columns
    return df


def create_dataset(path, num_classes, deletedups=False, randomize=False, drop_digits=6, include_raw=False,
                   group_fingers=False,
                   label_index=0):
    import pandas as pd
    import numpy as np
    from keras.utils import np_utils
    import os
    import re

    # sensors = ['id', 'thu-near', 'thu-far', 'thu-ind', 'ind-near', 'ind-far', 'ind-mid', 'mid-near', 'mid-far',
    #            'mid-rin', 'rin-near', 'rin-far', 'rin-lil', 'lil-near', 'lil-far', 'yaw', 'pitch', 'roll']

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

    if include_raw:
        raw_path = re.sub('scaled', 'raw', path)
        raw_data = create_dataset(path=raw_path, deletedups=deletedups, randomize=randomize,
                                  group_fingers=group_fingers)

        dataset = dataset.merge(raw_data, how='inner', left_index=True, right_index=True)
        dataset = dataset[
            ['id', 'thu-near_x', 'thu-near_y', 'thu-far_x', 'thu-far_y', 'thu-ind_x', 'thu-ind_y', 'ind-near_x',
             'ind-near_y',
             'ind-far_x', 'ind-far_y', 'ind-mid_x', 'ind-mid_y', 'mid-near_x', 'mid-near_y', 'mid-far_x', 'mid-far_y',
             'mid-rin_x', 'mid-rin_y', 'rin-near_x', 'rin-near_y', 'rin-far_x', 'rin-far_y', 'rin-lil_x', 'rin-lil_y',
             'lil-near_x', 'lil-near_y', 'lil-far_x', 'lil-far_y', 'yaw_x', 'yaw_y', 'pitch_x', 'pitch_y', 'roll_x',
             'roll_y']]

    if deletedups:
        dataset = dataset.drop_duplicates().reset_index(drop=True)

    # TODO: This randomization has to be stratified
    if randomize:
        dataset = dataset.sample(frac=1).reset_index(drop=True)

    if drop_digits:
        dataset = dataset.round(drop_digits)

    if group_fingers:
        dataset = dataset[['id', 'thu-ind', 'ind-mid', 'mid-rin', 'rin-lil', 'thu-near', 'thu-far', 'ind-near',
                           'ind-far', 'mid-near', 'mid-far', 'rin-near', 'rin-far', 'lil-near', 'lil-far', 'yaw',
                           'pitch', 'roll']]

    y, x = np.hsplit(dataset, [1])
    y = y - label_index

    x = np.expand_dims(x, axis=2)
    y = np_utils.to_categorical(y, num_classes)
    return x, y


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
    from keras.activations import softmax, relu

    DROPOUT_RATE = 0.3
    ACTIVATION_FUNCTION = relu
    output_size = 8
    feature_map_size = 256

    inputs = Input(shape=(input_dim, 1))

    bn_input = BatchNormalization()(inputs)

    conv_1 = Conv1D(output_size, 1, activation=ACTIVATION_FUNCTION, kernel_regularizer=l1_l2(0.01))(bn_input)
    conv_1 = Dropout(DROPOUT_RATE)(conv_1)

    conv_2 = Conv1D(output_size, 1, activation=ACTIVATION_FUNCTION, kernel_regularizer=l1_l2(0.01))(bn_input)
    conv_2 = Dropout(DROPOUT_RATE)(conv_2)
    conv_2 = Conv1D(output_size, 3, activation=ACTIVATION_FUNCTION, kernel_regularizer=l1_l2(0.01))(conv_2)
    conv_2 = Dropout(DROPOUT_RATE)(conv_2)
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
    from keras.activations import softmax, selu

    # TODO: Consider 'same', 'valid' options for padding

    DROPOUT_RATE = 0.4
    ACTIVATION_FUNCTION = selu
    output_size = 16
    pool_size = 3
    feature_map_size = 100

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

    pool_1 = MaxPool1D(pool_size, padding='same')(bn_input)
    conv_4 = Conv1D(output_size, 1, activation=ACTIVATION_FUNCTION, kernel_regularizer=l1_l2(0.01))(pool_1)
    conv_4 = Dropout(DROPOUT_RATE)(conv_4)

    output = concatenate([conv_1, conv_2, conv_3, conv_4], axis=1)

    flatten = Flatten()(output)
    dense = Dense(feature_map_size, activation=ACTIVATION_FUNCTION, kernel_regularizer=l1_l2(0.01))(flatten)
    dense = Dropout(DROPOUT_RATE)(dense)
    output = Dense(output_dim, activation=softmax)(dense)

    model = Model(inputs=inputs, outputs=output)
    return model


def split_functional(input_dim, output_dim):
    from keras.models import Model
    from keras.regularizers import l1_l2
    from keras.layers import Conv1D, Flatten, Dropout, Dense, Input, concatenate, BatchNormalization, Lambda
    from keras.activations import softmax, selu

    def slice_before(t):
        return t[:, :4, :]

    def slice_after(t):
        return t[:, 4:, :]

    DROPOUT_RATE = 0.5
    ACTIVATION_FUNCTION = selu
    output_size = 16
    feature_map_size = 50

    inputs = Input(shape=(input_dim, 1))

    knuckles = Lambda(slice_before)(inputs)
    fingers = Lambda(slice_after)(inputs)

    print(fingers.shape)
    print(knuckles.shape)

    bn_fingers = BatchNormalization()(fingers)
    bn_knuckles = BatchNormalization()(knuckles)

    reconcat = concatenate([bn_fingers, bn_knuckles], axis=1)
    conv = Conv1D(output_size, 1, activation=ACTIVATION_FUNCTION, kernel_regularizer=l1_l2())(reconcat)

    conv_knuckles = Conv1D(output_size, 1, activation=ACTIVATION_FUNCTION, kernel_regularizer=l1_l2())(bn_knuckles)
    conv_knuckles = Dropout(DROPOUT_RATE)(conv_knuckles)
    conv_knuckles = Conv1D(output_size, 2, activation=ACTIVATION_FUNCTION, kernel_regularizer=l1_l2())(conv_knuckles)
    conv_knuckles = Dropout(DROPOUT_RATE)(conv_knuckles)
    conv_knuckles = Conv1D(output_size, 2, activation=ACTIVATION_FUNCTION, kernel_regularizer=l1_l2())(conv_knuckles)
    conv_knuckles = Dropout(DROPOUT_RATE)(conv_knuckles)

    conv_fingers = Conv1D(output_size, 1, activation=ACTIVATION_FUNCTION, kernel_regularizer=l1_l2())(bn_fingers)
    conv_fingers = Dropout(DROPOUT_RATE)(conv_fingers)
    conv_fingers = Conv1D(output_size, 2, activation=ACTIVATION_FUNCTION, kernel_regularizer=l1_l2())(conv_fingers)
    conv_fingers = Dropout(DROPOUT_RATE)(conv_fingers)
    conv_fingers = Conv1D(output_size, 4, activation=ACTIVATION_FUNCTION, kernel_regularizer=l1_l2())(conv_fingers)
    conv_fingers = Dropout(DROPOUT_RATE)(conv_fingers)

    concat = concatenate([conv_knuckles, conv_fingers, conv], axis=1)

    flatten = Flatten()(concat)
    dense = Dense(feature_map_size, activation=ACTIVATION_FUNCTION, kernel_regularizer=l1_l2())(flatten)
    dense = Dropout(DROPOUT_RATE)(dense)
    output = Dense(output_dim, activation=softmax)(dense)

    model = Model(inputs=inputs, outputs=output)
    return model


def last_to_first():
    df = create_dataset('datasets/old_datasets/dataset_marcelo_RG.txt', deletedups=False, randomize=False)
    df.columns = ['thu-near', 'thu-far', 'thu-ind', 'ind-near', 'ind-far', 'ind-mid', 'mid-near', 'mid-far',
                  'mid-rin', 'rin-near', 'rin-far', 'rin-lil', 'lil-near', 'lil-far', 'yaw', 'pitch', 'roll', 'id']

    cols = df.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    df = df[cols]

    df.to_csv('marcelo-r-scaled.txt', sep='\t', header=False, index=False, float_format='%.3f')
