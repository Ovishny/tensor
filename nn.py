import tensorflow as tf 
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt 

#load in dataset
fashion_mnist = keras.datasets.fashion_mnist 
#split data into testing and training
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

#preprocess to fit pixel values between 0 and 1
train_images = train_images/255.0
test_images = test_images/255.0

#building the model 
model = keras.Sequential([
	keras.layers.Flatten(input_shape = (28,28)),#input layer 
	keras.layers.Dense(128, activation = 'relu'),#hidden layer
	keras.layers.Dense(10, activation = 'softmax')#output layer
	])