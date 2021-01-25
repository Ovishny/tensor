import tensorflow as tf 
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt 

#load in dataset
fashion_mnist = keras.datasets.fashion_mnist 
#split data into testing and training
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
