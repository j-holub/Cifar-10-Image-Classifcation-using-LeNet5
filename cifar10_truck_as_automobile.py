# Keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten
from keras.utils.np_utils import to_categorical
from keras import optimizers

from keras.datasets import cifar10

# NumPy
import numpy as np

# Python Std Lib
import os

# User Lib
import lib.plot as plot

# get the training and test data
(input_train, output_train), (input_test, output_test) = cifar10.load_data()

# relabel class 10 (Truck) as class 2 (Automobile)
# train
order_index_set_train = np.argsort(output_train, axis=0)
output_train[order_index_set_train[45000:]] = [1]
# test
order_index_set_test = np.argsort(output_test, axis=0)
output_test[order_index_set_test[9000:]] = [1]


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
model.add(Dense(9, activation='softmax'))

# compile the model
model.compile(optimizer=optimizers.sgd(lr=0.01, momentum=0.9),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# train the model
history = model.fit(input_train, to_categorical(output_train), epochs=10, batch_size=32)

# test
score = model.evaluate(input_test, to_categorical(output_test), batch_size=32)

# print test set results
print("Testset Loss: %f" % score[0])
print("Testset Accuracy: %f" % score[1])

# Plot the history
os.makedirs(os.path.join(os.path.dirname(os.path.realpath(__file__)), "plots"), exist_ok=True)
plot.plot_training_loss(history, show=False, save_file="plots/cifar9_truck_as_automobile_loss.png")
plot.plot_training_accuracy(history, show=False, save_file="plots/cifar9_truck_as_automobile_accuracy.png")
