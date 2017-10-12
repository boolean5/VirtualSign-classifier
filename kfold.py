import argparse

import numpy as np
from keras.losses import categorical_crossentropy
from keras.utils import np_utils
from sklearn.model_selection import KFold

from utils import *

SENSORS = 14
BATCH_SIZE = 50
EPOCHS = 150
NUM_CLASSES = 42    # TODO: Get this from the data

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

# Parsing from terminal
parser = argparse.ArgumentParser(description='K-Fold validation script')
parser.add_argument('dataset_path', help='Path of datasets folder')
parser.add_argument('-m', '--model', help='Choose model to train on from [inception, seq_v1, seq_v2, functional]',
                    type=str, default='inception')
parser.add_argument('-k', '--kfolds', type=int, help='Number of folds', default=10)
args = parser.parse_args()

dataset_path = args.dataset_path
model_type = args.model
k = args.kfolds

# Data loading
# TODO: Tackle weird float conversion from pandas to numpy array
dataset = create_dataset(dataset_path).as_matrix()
x, y = np.hsplit(dataset, [-1])
x = np.expand_dims(x, axis=2)
y = np_utils.to_categorical(y, NUM_CLASSES)

# K-fold cross validation
kfold = KFold(n_splits=k, shuffle=True, random_state=seed)
cvscores = []
for train, test in kfold.split(x, y):
    # Model building
    model = build_model(model_type, SENSORS, NUM_CLASSES)
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
