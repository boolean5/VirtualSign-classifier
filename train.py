import numpy as np
import pandas
from keras.models import Sequential
from keras.layers import Dense, Conv1D, MaxPool1D
from keras.activations import relu, tanh

# Hyper-parameters
sensors = 14
batch_size = 10

model = Sequential()

# TODO decide structure of CNN
model.add(Conv1D(10, 3, input_shape=[sensors, batch_size]))
model.add(MaxPool1D(pool_size=2)
