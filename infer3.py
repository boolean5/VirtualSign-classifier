import sys
import argparse
import numpy as np
from keras.models import load_model

file = open("testfile", "w")
file.write("Python script invoked")

parser = argparse.ArgumentParser(description='Classify a hand configuration sequence')
parser.add_argument('model_path', help='Path of the model to be used for evaluation')
args = parser.parse_args()

model_path = args.model_path

# returns a compiled model identical to the previous one
model = load_model(model_path)

for line in sys.stdin:
    input_array = list(map(float, line.split()))
    input_array = np.asarray(input_array)
    input_array = np.expand_dims(input_array, axis=0)
    if input_array.shape[1] == 14:
        input_array = np.expand_dims(input_array, axis=2)
        output = model.predict(input_array, batch_size=1, verbose=0)
        output = output.flatten()

        # uncomment this section to get the top 3 more probable gestures
        # along with their respective probabilities
        gestures = np.argpartition(output, -3)[-3:]
        probabilities = output[gestures]
        gestures = gestures[np.argsort(probabilities)]
        probabilities.sort()
        for i in range (2,-1,-1):
            print(gestures[i], probabilities[i])

        # uncomment this section to get only the most probable gesture
        '''probability = max(output)
        gesture = np.where(output == probability)[0]
        # print(int(gesture))
        print(int(gesture), probability)'''

        sys.stdout.flush()

file.write("Found EOF")
file.close()
