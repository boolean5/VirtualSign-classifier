import numpy as np
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam
from keras.utils import np_utils
from utils import *
from sklearn.model_selection import KFold

SENSORS = 14
BATCH_SIZE = 50
EPOCHS = 150
NUM_CLASSES = 42    # TODO: Get this from the data

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

# Data loading
# TODO: Tackle weird float conversion from pandas to numpy array
dataset = create_dataset('datasets/').as_matrix()
x, y = np.hsplit(dataset, [-1])
x = np.expand_dims(x, axis=2)
y = np_utils.to_categorical(y, NUM_CLASSES)

# K-fold cross validation
kfold = KFold(n_splits=10, shuffle=True, random_state=seed)
cvscores = []
for train, test in kfold.split(x, y):
    # create model
    model = build_functional(SENSORS, NUM_CLASSES)
    # compile model
    model.compile(loss=categorical_crossentropy, optimizer='adam', metrics=['accuracy'])
    # fit model
    model.fit(x[train], y[train], batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=0, shuffle=True)
    # evaluate model
    scores = model.evaluate(x[test], y[test], verbose=0)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    cvscores.append(scores[1] * 100)


print(model.summary())
print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))
