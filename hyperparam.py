import os

from hyperas import optim
from hyperas.distributions import uniform, choice
from hyperopt import Trials, STATUS_OK, tpe

from utils import create_dataset


def functional(x_train, y_train, x_dev, y_dev):
    from keras.models import Model
    from keras.regularizers import l1_l2
    from keras.layers import Conv1D, Flatten, Dropout, Dense, Input, concatenate, BatchNormalization
    from keras.activations import softmax, selu, relu
    from keras.losses import categorical_crossentropy
    from keras.callbacks import EarlyStopping

    # Hyper-parameters
    SENSORS = len(x_train[0])
    BATCH_SIZE = 64
    EPOCHS = 200
    NUM_CLASSES = 42  # TODO: Get this from the data

    DROPOUT_RATE = {{uniform(0, 0.5)}}
    ACTIVATION_FUNCTION = {{choice([selu, relu])}}
    output_size = {{choice([4, 8, 16, 32, 64])}}
    feature_map_size = {{choice([64, 128, 256])}}

    inputs = Input(shape=(SENSORS, 1))

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
    output = Dense(NUM_CLASSES, activation=softmax)(dense)

    model = Model(inputs=inputs, outputs=output)
    model.compile(loss=categorical_crossentropy, optimizer='adam', metrics=['categorical_accuracy'])

    callbacks = [EarlyStopping(patience=20)]
    model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=0, validation_data=(x_dev, y_dev),
              shuffle=True, callbacks=callbacks)

    loss, acc = model.evaluate(x_dev, y_dev, batch_size=1)

    print('Test score:', loss)
    print('Test accuracy:', acc)
    return {'loss': -acc, 'status': STATUS_OK, 'model': model}


def data():
    x_train, y_train = create_dataset('datasets/new_datasets/scaled/right', 42, label_index=1)
    x_dev, y_dev = create_dataset('datasets/testing/', 42)
    return x_train, y_train, x_dev, y_dev


if __name__ == '__main__':
    # Turn off TensorFlow warnings
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    best_run, best_model = optim.minimize(model=functional,
                                          data=data,
                                          algo=tpe.suggest,
                                          max_evals=2,
                                          trials=Trials())
    X_train, Y_train, X_test, Y_test = data()
    print("Evalutation of best performing model:")
    print(best_model.evaluate(X_test, Y_test))
    print(best_run)
