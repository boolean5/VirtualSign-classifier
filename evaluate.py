import sys
import numpy as np
from keras.models import load_model
from keras.utils import np_utils
from utils import create_dataset

model_path = sys.argv[1]
model = load_model(model_path)

data_path = sys.argv[2]
test_set = create_dataset(data_path, randomize=False)

print(test_set.shape)

x_test, y_test = np.hsplit(test_set, [-1])
y_test = np_utils.to_categorical(y_test, 42)
x_test = np.expand_dims(x_test, axis=2)
print(x_test.shape)

loss, acc = model.evaluate(x_test, y_test, verbose=1)
print(loss, acc)
