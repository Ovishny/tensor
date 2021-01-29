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

#use to look at images in dataset--------------------------------
# IMG_INDEX = 1 #change this to look at different images

# plt.imshow(train_images[IMG_INDEX], cmap=plt.cm.binary)
# plt.xlabel(class_names[train_labels[IMG_INDEX][0]])
# plt.show()
# -----------------------------------------------------------------

#cnn architecture. A stack of convolution 2D layers and Max Pooling 2D layers followed by a few densely connected layers
model = models.Sequential()
model.add(layers.Conv2D(32,(3,3), activation = 'relu', input_shape = (32,32,3)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64,(3,3), activation = 'relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64,(3,3), activation = 'relu'))

# #view a summary of the model--------------------------
# print(model.summary())
# #-------------------------------------------------

#adding dense layers. These layers act as the classifiers for the model
model.add(layers.Flatten())#makes 1D
model.add(layers.Dense(64,activation = 'relu'))
model.add(layers.Dense(10))#for amount of classes

#training
model.compile(optimizer = 'adam',
	loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True),
	metrics = ['accuracy'])
history = model.fit(train_images, train_labels, epochs = 10, 
	validation_data = (test_images, test_labels))

#Evaluating the model