import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Conv1D, MaxPool1D, Flatten, Dropout
from keras.activations import relu, softmax
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam
from keras.utils import np_utils
from utils import create_dataset
from keras.callbacks import ModelCheckpoint
import h5py

# Hyper-parameters
# TODO: Distinguish between hyper-parameters and training parameters
# TODO: Hyper-parameters are usually capitalized
SENSORS = 14
BATCH_SIZE = 10
EPOCHS = 100
filter_size_1 = 2
filter_size_2 = 5
output_size_1 = 30
output_size_2 = 20
pool_size_1 = 2
pool_size_2 = 1
feature_map_size = 200

num_classes = 42    # TODO: Get this from the data

# Data loading
# TODO: Tackle weird float conversion from pandas to numpy array
dataset = create_dataset('datasets/').as_matrix()
x, y = np.hsplit(dataset, [-1])

# Data pre-processing
# TODO: Implement train-val-test split
splitPoint = int(np.ceil(len(y) * 0.75))
x_train, x_val = np.vsplit(x, [splitPoint])
y_train, y_val = np.vsplit(y, [splitPoint])

y_train = np_utils.to_categorical(y_train, num_classes)
y_val = np_utils.to_categorical(y_val, num_classes)
x_train = np.expand_dims(x_train, axis=2)
x_val = np.expand_dims(x_val, axis=2)

# Model building
# TODO: Add multiple & varied filters: https://github.com/fchollet/keras/issues/1023
model = Sequential()
model.add(Conv1D(20, filter_size_1, input_shape=(SENSORS, 1), activation=relu))

# TODO: Consider changing strides from None(which defaults to pool size)
model.add(MaxPool1D(pool_size=pool_size_1, strides=1))
# model.add(Dropout(0.5))
model.add(Conv1D(10, filter_size_2, activation=relu))
model.add(MaxPool1D(pool_size_2, strides=1))
# model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(feature_map_size, activation=relu))
model.add(Dense(num_classes, activation=softmax))

model.compile(loss=categorical_crossentropy,
              optimizer='adam',
              metrics=['accuracy'])

model.summary()

callbacks = [ModelCheckpoint('model.hdf5', monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='min', period=1)]

# TODO: Check callbacks to history objects
hist = model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=1, validation_data=(x_val, y_val), callbacks=callbacks)

min_val_loss_epoch = min(range(len(hist.history['val_loss'])), key=hist.history['val_loss'].__getitem__)
min_val_loss = hist.history['val_loss'][min_val_loss_epoch]
print('Minimum validation loss of %.4f at epoch %d.' % (min_val_loss, min_val_loss_epoch+1))

