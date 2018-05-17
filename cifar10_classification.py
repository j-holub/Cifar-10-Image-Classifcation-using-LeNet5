# Network
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Dense

# Data
from keras.datasets import cifar10


# get the training and test data
(input_train, output_train), (input_test, output_test) = cifar10.load_data()

# creating the basic model
model = Sequential()
