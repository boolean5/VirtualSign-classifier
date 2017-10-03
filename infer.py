import sys
import numpy as np
from keras.models import load_model

# the first argument should be the file in which the pre-trained model is stored
# the second argument should be the vector containing the 14 sensor values
filename = sys.argv[1]

# returns a compiled model identical to the previous one
model = load_model(filename)

input_array = np.asarray(sys.argv)
input_array = np.delete(input_array, [0, 1])

input_array = input_array.reshape(1,14,1)

output = model.predict(input_array, batch_size=1, verbose=0)
#output = model.predict_classes(input_array)
output = output.flatten()
probability = max(output)
gesture = np.where(output==probability)[0]

print(int(gesture), probability)
