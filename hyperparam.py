import os

from hyperas import optim
from hyperas.distributions import uniform, choice
from hyperopt import Trials, STATUS_OK, tpe

from utils import create_dataset


def functional(x_train, y_train, x_dev, y_dev):
    from keras.activations import selu, relu
    from keras.losses import categorical_crossentropy
    from keras.callbacks import EarlyStopping
    from utils import build_functional

    # Hyper-parameters
    SENSORS = len(x_train[0])
    BATCH_SIZE = {{choice([16, 32, 64, 128])}}
    EPOCHS = 200
    NUM_CLASSES = 42  # TODO: Get this from the data

    DROPOUT_RATE = {{uniform(0, 0.5)}}
    ACTIVATION_FUNCTION = {{choice([selu, relu])}}
    output_size = {{choice([4, 8, 16, 32, 64])}}
    feature_map_size = {{choice([64, 128, 256])}}
    reg_l1 = {{choice([0, 0.003, 0.01, 0.03])}}
    reg_l2 = {{choice([0, 0.003, 0.01, 0.03])}}

    model = build_functional(SENSORS, NUM_CLASSES, dropout_rate=DROPOUT_RATE, output_size=output_size, reg_1=reg_l1,
                             reg_2=reg_l2, feature_map_size=feature_map_size, activation_function=[ACTIVATION_FUNCTION])

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
                                          max_evals=120,
                                          trials=Trials())
    X_train, Y_train, X_test, Y_test = data()
    print("Evalutation of best performing model:")
    print(best_model.evaluate(X_test, Y_test))
    print(best_run)
