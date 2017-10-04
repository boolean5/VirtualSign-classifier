import numpy as np
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam
from keras.utils import np_utils
from utils import create_dataset, build_sequential_v1, build_inception_like
from keras.callbacks import ModelCheckpoint, TensorBoard
import h5py

# Hyper-parameters
# TODO: Distinguish between hyper-parameters and training parameters
SENSORS = 14
BATCH_SIZE = 20
EPOCHS = 100
NUM_CLASSES = 42    # TODO: Get this from the data

# Data loading
# TODO: Tackle weird float conversion from pandas to numpy array
dataset = create_dataset('datasets/').as_matrix()
x, y = np.hsplit(dataset, [-1])

# Data pre-processing
# TODO: Implement train-val-test split
splitPoint = int(np.ceil(len(y) * 0.75))
x_train, x_val = np.vsplit(x, [splitPoint])
y_train, y_val = np.vsplit(y, [splitPoint])

y_train = np_utils.to_categorical(y_train, NUM_CLASSES)
y_val = np_utils.to_categorical(y_val, NUM_CLASSES)
x_train = np.expand_dims(x_train, axis=2)
x_val = np.expand_dims(x_val, axis=2)

# Model building
model = build_inception_like(SENSORS, NUM_CLASSES)

model.compile(loss=categorical_crossentropy,
              optimizer='adam',
              metrics=['accuracy'])

model.summary()

callbacks = [ModelCheckpoint('model.hdf5',
                             monitor='val_loss',
                             verbose=0,
                             save_best_only=True,
                             save_weights_only=False,
                             mode='min',
                             period=1),

             TensorBoard(log_dir='./logs',
                         histogram_freq=1,
                         batch_size=10,
                         write_graph=True,
                         write_grads=True,
                         write_images=True)]


hist = model.fit(x_train,
                 y_train,
                 batch_size=BATCH_SIZE,
                 epochs=EPOCHS, verbose=1,
                 validation_data=(x_val, y_val),
                 # shuffle=True,
                 callbacks=callbacks)

min_val_loss_epoch = min(range(len(hist.history['val_loss'])), key=hist.history['val_loss'].__getitem__)
min_val_loss = hist.history['val_loss'][min_val_loss_epoch]
print('Minimum validation loss of %.4f at epoch %d.' % (min_val_loss, min_val_loss_epoch+1))
