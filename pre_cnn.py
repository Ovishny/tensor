#using a pretrained model for a more accurate cnn

#imports
import os
import numpy as np 
import matplotlib.pyplot as plt 
import tensorflow as tf 
keras = tf.keras

#dataset
import tensorflow_datasets as tfds 
tfds.disable_progress_bar()

#split the data manually into 80% training, 10% testing, 10% validation
(raw_train, raw_validation, raw_test), metadata = tfds.load(
	'cats_vs_dogs',
	split = ['train[:80%]', 'train[80%:90%]', 'train[90%:]'],
	with_info = True,
	as_supervised = True,
	)

get_label_name = metadata.features['label'].int2str #creates a function object that we can use to get labels



IMG_SIZE = 160 #All images will be resized to 160x160

#return image that is reshaped to IMG_SIZE
def format_example(image, label):
	image = tf.cast(image, tf.float32)
	image = (image/127.5) - 1
	image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
	return image, label

train = raw_train.map(format_example)
validation = raw_validation.map(format_example)
test = raw_test.map(format_example)

#display 2 images from the dataset
for image,label in raw_train.take(2):
	plt.figure()
	plt.imshow(image)
	plt.title(get_label_name(label))