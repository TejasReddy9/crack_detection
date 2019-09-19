print("Importing libraries...")

import tensorflow as tf
from tensorflow.keras.models import Sequential, model_from_json
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
import numpy as np
import h5py
import datetime as dt
import os


img_size = 128

print("Loading the data...")
hf = h5py.File('./concrete_crack_image_data.h5', 'r') #Replace the three dots with the directory you saved the dataset in
X = np.array(hf.get('X_concrete'))
y = np.array(hf.get("y_concrete"))
hf.close()
print("Data successfully loaded!")

print("Scaling the data...!")
X = X / 255
print("Data successfully scaled!")

model = Sequential()

model.add(Conv2D(16, (3, 3), activation = "relu", input_shape = (img_size, img_size, 1)))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(.3))

model.add(Conv2D(32, (3, 3), activation = "relu"))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(.3))

model.add(Conv2D(32, (3, 3), activation = "relu"))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(.3))

model.add(Flatten())
model.add(Dense(258, activation = "relu"))

model.add(Dense(1, activation = "sigmoid"))

print("Compiling the model...")
model.compile(loss = "binary_crossentropy", optimizer = "adam", metrics = ["accuracy"])
print("Model successfully compiled!!")

print("Fitting the model...")
model.fit(X, y, batch_size = 64, epochs = 5, validation_split = .2)
print("Model successfully fitted!!")

print("Saving the model...")
model.save(os.path.join(os.getcwd(), str(dt.datetime.now()).split()[0] + str(dt.datetime.now()).split()[1], "/crack_classifier.model")) #Replace the dots with the directory you want to save the model in
print("Model successfully saved!!")

print("Saving weights and model json...")
model_json = model.to_json()
with open("model.json","w") as json_file:
    json_file.write(model_json)
model.save_weights("model.h5")
print("Weights and model Json saved!!")