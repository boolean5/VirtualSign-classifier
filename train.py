import argparse
import os

import numpy as np
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.losses import categorical_crossentropy
from keras.utils import np_utils

from utils import *

# Hyper-parameters
SENSORS = 14
BATCH_SIZE = 50
EPOCHS = 200
NUM_CLASSES = 42  # TODO: Get this from the data

# Parsing from terminal
parser = argparse.ArgumentParser(description='Train a hand configuration classifier')
parser.add_argument('dataset_path', help='Path of datasets folder')
parser.add_argument('-m', '--model', help='Choose model to train on from [inception, seq_v1, seq_v2, functional]',
                    type=str, default='inception')
parser.add_argument('-e', '--epochs', help='Number of training epochs', type=int, default=EPOCHS)
parser.add_argument('-b', '--batch', help='Size of training batch', type=int, default=BATCH_SIZE)
args = parser.parse_args()

dataset_path = args.dataset_path
model_type = args.model
EPOCHS = args.epochs  # Epochs and batch size are assigned twice which is obsolete. This will change depending on
BATCH_SIZE = args.batch  # how the hyper-parameter search script is called. Leaving as is for now.

# Data loading
# TODO: Tackle weird float conversion from pandas to numpy array
dataset = create_dataset(dataset_path, deletedups=False).as_matrix()

# Data pre-processing
y, x = np.hsplit(dataset, [1])
splitPoint = int(np.ceil(len(y) * 0.9))
x_train, x_val = np.vsplit(x, [splitPoint])
y_train, y_val = np.vsplit(y, [splitPoint])

# I add these lines for our own datasets because they range from 1 to 42
# y_train = y_train - 1
# y_val = y_val - 1

y_train = np_utils.to_categorical(y_train, NUM_CLASSES)
y_val = np_utils.to_categorical(y_val, NUM_CLASSES)
x_train = np.expand_dims(x_train, axis=2)
x_val = np.expand_dims(x_val, axis=2)

# Model building
model = build_model(model_type, SENSORS, NUM_CLASSES)

model.compile(loss=categorical_crossentropy,
              optimizer='adam',
              metrics=['accuracy'])
model.summary()

callbacks = [ModelCheckpoint('saved_models/' + model_type + '.hdf5',
                             monitor='val_loss',
                             verbose=0,
                             save_best_only=True,
                             save_weights_only=False,
                             mode='min',
                             period=1),
             TensorBoard(log_dir='./logs',
                         histogram_freq=1,
                         batch_size=BATCH_SIZE,
                         write_graph=True,
                         write_grads=True,
                         write_images=True)]

hist = model.fit(x_train,
                 y_train,
                 batch_size=BATCH_SIZE,
                 epochs=EPOCHS, verbose=1,
                 validation_data=(x_val, y_val),
                 shuffle=True,
                 callbacks=callbacks)

min_val_loss_epoch = min(range(len(hist.history['val_loss'])), key=hist.history['val_loss'].__getitem__)
min_val_loss = hist.history['val_loss'][min_val_loss_epoch]
print('Minimum validation loss of {:.4f} at epoch {}.'.format(min_val_loss, min_val_loss_epoch + 1))

os.rename('saved_models/{}.hdf5'.format(model_type),
          'saved_models/{}-{:.4f}-{:0>3}.hdf5'.format(model_type, min_val_loss, min_val_loss_epoch + 1))
