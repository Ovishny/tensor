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
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose = 2)
print(test_acc)

#DATA AUGMENTATION(CREATING MULTIPLE AUGMENTS OF IMAGES IN THE DATA SET SO THE MODEL HAS MORE IMAGES TO WORK WITH, MAKING THE MODEL MORE ACCURATE)
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator

#creates a data generator object that transforms images
datagen = ImageDataGenerator(
	rotation_range = 40,
	width_shift_range = 0.2,
	height_shift_range = 0.2,
	shear_range = 0.2,
	zoom_range = 0.2,
	horizontal_flip = True,
	fill_mode = 'nearest')

#pick an image to transform
test_img = train_images[14]
img = image.img_to_array(test_img)#convert image to numpy array
img = img.reshape((1,) + img.shape)#reshape image

i = 0

for batch in datagen.flow(img, save_prefix = 'test', save_format = 'jpeg'): #this loops runs forever until we break, saving images to current directory
	plt.figure(i)
	plot = plt.imshow(image.img_to_array(batch[0]))
	i += 1
	if i > 4: #show 4 images
		break

plt.show()

