import argparse
import numpy as np
from keras.models import load_model
from keras.utils import np_utils
from utils import create_dataset

# Parsing from terminal
parser = argparse.ArgumentParser(description='Evaluate a trained model')
parser.add_argument('test_set_path', help='Path of test set folder')
parser.add_argument('model_path', help='Path of the model to be evaluated')
args = parser.parse_args()

test_set_path = args.test_set_path
model_path = args.model_path

# Loading
model = load_model(model_path)
test_set = create_dataset(test_set_path, randomize=False)

# Data manipulation
x_test, y_test = np.hsplit(test_set, [-1])
y_test = np_utils.to_categorical(y_test, 42)
x_test = np.expand_dims(x_test, axis=2)
print(x_test.shape)

# Evaluation
loss, acc = model.evaluate(x_test, y_test, verbose=1)
print(loss, acc)
