import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Conv1D, MaxPool1D, Flatten
from keras.activations import relu, softmax
from keras.losses import categorical_crossentropy
from keras.optimizers import SGD, Adam
from utils import create_dataset

# Hyper-parameters
# TODO: Distinguish between hyperparameters and training parameters
# TODO: Hyperparameters are usually capitalized
SENSORS = 14
BATCH_SIZE = 10
EPOCHS = 20
filter_size_1 = filter_size_2 = 4
output_size_1 = output_size_2 = 10
pool_size_1 = 2
pool_size_2 = 1

feature_map_size = 120
num_classes = 42    # We can draw this from the dataset

# Data loading
# TODO: Tackle weird float conversion from pandas to numpy array
dataset = create_dataset('datasets/').as_matrix()
x, y = np.hsplit(dataset, [-1])
print(x.shape, y.shape)
splitPoint = int(np.ceil(len(y) * 0.75))
print('splitpoint is:', splitPoint)
x_train, x_val = np.vsplit(x, [splitPoint])
y_train, y_val = np.vsplit(y, [splitPoint])
print(x_train.shape, x_val.shape)
print(y_train.shape, y_val.shape)

# Model building
# TODO: Add multiple & varied filters: https://github.com/fchollet/keras/issues/1023
# TODO: Check if output shape is what i think it is (first argument)
# TODO: Add dropout
model = Sequential()
model.add(Conv1D(20, filter_size_1, input_shape=(SENSORS, 1), activation=relu))

# TODO: Consider changing strides from None(which defaults to pool size)
model.add(MaxPool1D(pool_size=pool_size_1, strides=1))

model.add(Conv1D(10, filter_size_2, activation=relu))
model.add(MaxPool1D(pool_size_2, strides=None))
model.add(Flatten())
model.add(Dense(feature_map_size, activation=relu))
model.add(Dense(num_classes, activation=softmax))

model.compile(loss=categorical_crossentropy,
              optimizer='adam',
              metrics=['accuracy'])

model.summary()

# TODO: Check callbacks to history objects
model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=1, validation_data=(x_val, y_val))