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