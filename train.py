import argparse
import os

import numpy as np
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.losses import categorical_crossentropy
from keras.utils import np_utils

from utils import *

# Turn off TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Hyper-parameters
SENSORS = 17
BATCH_SIZE = 64
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

# Data loading & pre-processing
training_set = create_dataset(dataset_path, randomize=False)
dev_test_set = create_dataset('datasets/testing/', randomize=False)  # TODO: Make this randomization stratified

y_train, x_train = np.hsplit(training_set, [1])
y_dev_test, x_dev_test = np.hsplit(dev_test_set, [1])

splitPoint = int(np.ceil(len(y_dev_test) * 0.5))
x_dev, x_test = np.vsplit(x_dev_test, [splitPoint])
y_dev, y_test = np.vsplit(y_dev_test, [splitPoint])

# I add these lines for our own datasets because they range from 1 to 42. This is not permanent
y_train = y_train - 1
# y_dev = y_dev - 1
# y_test = y_test - 1

x_train = np.expand_dims(x_train, axis=2)
x_dev = np.expand_dims(x_dev, axis=2)
x_test = np.expand_dims(x_test, axis=2)
y_train = np_utils.to_categorical(y_train, NUM_CLASSES)
y_dev = np_utils.to_categorical(y_dev, NUM_CLASSES)
y_test = np_utils.to_categorical(y_test, NUM_CLASSES)

# Model building
model = build_model(model_type, SENSORS, NUM_CLASSES)

model.compile(loss=categorical_crossentropy, optimizer='adam', metrics=['categorical_accuracy'])
model.summary()

callbacks = [ModelCheckpoint('saved_models/' + model_type + '.hdf5',
                             monitor='val_loss',
                             verbose=0,
                             save_best_only=True,
                             save_weights_only=False,
                             mode='min',
                             period=1),
             # TensorBoard(log_dir='./logs',
             #             histogram_freq=1,
             #             batch_size=BATCH_SIZE,
             #             write_graph=True,
             #             write_grads=True,
             #             write_images=True),
             EarlyStopping(patience=10)]

# Training
hist = model.fit(x_train,
                 y_train,
                 batch_size=BATCH_SIZE,
                 epochs=EPOCHS, verbose=0,
                 validation_data=(x_dev, y_dev),
                 shuffle=True,
                 callbacks=callbacks)

min_val_loss_epoch = min(range(len(hist.history['val_loss'])), key=hist.history['val_loss'].__getitem__)
min_val_loss = hist.history['val_loss'][min_val_loss_epoch]
min_acc = hist.history['categorical_accuracy'][min_val_loss_epoch]
print('Minimum validation loss of {:.4f} at epoch {} with accuracy of {:.2f}%.'.format(min_val_loss,
                                                                                       min_val_loss_epoch + 1,
                                                                                       min_acc * 100))

os.rename('saved_models/{}.hdf5'.format(model_type),
          'saved_models/{}-{:.4f}-{:0>3}.hdf5'.format(model_type, min_val_loss, min_val_loss_epoch + 1))

# Evaluation
loss, acc = model.evaluate(x_test, y_test, verbose=0)
print('Results on test set: {:.4f} loss, {:.2f} accuracy.'.format(loss, acc * 100))
