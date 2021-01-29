#convolutional neural network on keras dataset

import tensorflow as tf 

from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt 

#load and split data
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
#normalize pixel values between 0 and 1
train_images, test_images = train_images/255.0 , test_images/255.0
#class names in dataset
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
				'dog', 'frog', 'horse', 'ship', 'truck']