# Network
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten
from keras.utils.np_utils import to_categorical
from keras import optimizers


# Data
from keras.datasets import cifar10

import numpy as np

# get the training and test data
(input_train, output_train), (input_test, output_test) = cifar10.load_data()

# creating the basic model
model = Sequential()

# 30 Conv Layer
model.add(Conv2D(30, kernel_size=(3, 3), padding='valid', activation='relu', input_shape=(32, 32, 3)))
# 15 Max Pool Layer
model.add(MaxPool2D(pool_size=(2, 2), padding='valid'))
# 13 Conv Layer
model.add(Conv2D(13, kernel_size=(3,3), padding='valid', activation='relu'))
# 6 Max Pool Layer
model.add(MaxPool2D(pool_size=(2, 2), padding='valid'))
# Flatten the Layer for transitioning to the Fully Connected Layers
model.add(Flatten())
# 120 Fully Connected Layer
model.add(Dense(120, activation='relu'))
# 84 Fully Connected Layer
model.add(Dense(86, activation='relu'))
# 10 Output
model.add(Dense(10, activation='softmax'))

# compile the model
model.compile(optimizer=optimizers.sgd(lr=0.01, momentum=0.9),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# train the model
model.fit(input_train/255, to_categorical(output_train), epochs=10, batch_size=32)


# test
score = model.evaluate(input_test/255, to_categorical(output_test), batch_size=32)
print(score)
