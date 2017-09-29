import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Conv1D, MaxPool1D, Flatten
from keras.activations import relu, tanh
from utils import create_dataset

# Hyper-parameters
# TODO: Distinguish between hyperparameters and training parameters
# TODO: Add multiple & varied filters
sensors = 14
batch_size = 10
filter_size_1 = filter_size_2 = 3
output_size_1 = output_size_2 = 10
pool_size_1 = pool_size_2 = 3
feature_map_size = 120
num_classes = 42    # We can draw this from the dataset

# Data loading
dataset = create_dataset('datasets/', randomize=False).as_matrix()

# Model building
model = Sequential()

# TODO: Check if output shape is what i think it is (first argument)
# TODO: Add dropout
model.add(Conv1D(10, filter_size_1, input_shape=[sensors, batch_size], activation='relu'))

# TODO: Consider changing strides from None(which defaults to pool size)
model.add(MaxPool1D(pool_size=pool_size_1, strides=None))

model.add(Conv1D(10, filter_size_2, activation='relu'))
model.add(MaxPool1D(pool_size_2))
model.add(Flatten())
model.add(Dense(feature_map_size, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

