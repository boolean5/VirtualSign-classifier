import argparse

import numpy as np
from keras.models import load_model

parser = argparse.ArgumentParser(description='Classify a hand configuration sequence')
parser.add_argument('model_path', help='Path of the model to be used for evaluation')
args, input_array = parser.parse_known_args()

model_path = args.model_path

# returns a compiled model identical to the previous one
model = load_model(model_path)
input_array = np.asarray(input_array)
input_array = np.expand_dims(input_array, axis=0)
input_array = np.expand_dims(input_array, axis=2)

output = model.predict(input_array, batch_size=1, verbose=0)

output = output.flatten()
probability = max(output)
gesture = np.where(output == probability)[0]

print(int(gesture), probability)
