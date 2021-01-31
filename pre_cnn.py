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

# #display 2 images from the dataset
# for image,label in raw_train.take(2):
# 	plt.figure()
# 	plt.imshow(image)
# 	plt.title(get_label_name(label))
# plt.show()

#shuffle and batch images
BATCH_SIZE = 32
SHUFFLE_BUFFER_SIZE = 1000

train_batches = train.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
validation_batches = validation.batch(BATCH_SIZE)
test_batches = test.batch(BATCH_SIZE)

#picking a pretrained model
IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)

#create base model from pretrained model MobileNet V2. This model is trained on millions of imagges and has 1000 diff classes
#but we only want the convolution base of the model. We dont want to load the classification layer. We tell the model what input
#shape to expect and to use predetermined weights from imagent(googles dataset)
base_model = tf.keras.applications.MobileNetV2(input_shape = IMG_SHAPE,
	include_top = False,
	weights = 'imagenet')

# #to check the shape of the model
# for image, _ in train_batches.take(1):
# 	pass
# feature_batch = base_model(image)
# print(feature_batch.shape)

#freezing the base
base_model.trainable = False #no longer train model

#add classifier
global_average_layer = tf.keras.layers.GlobalAveragePooling2D() #pooling layer
prediction_layer = keras.layers.Dense(1) #prediction layer with single dense neuron

#combine these layers with model
model = tf.keras.Sequential([
	base_model,
	global_average_layer,
	prediction_layer
])

#training the model
base_learning_rate = 0.0001
model.compile(optimizer = tf.keras.optimizers.RMSprop(lr = base_learning_rate),
	loss = tf.keras.losses.BinaryCrossentropy(from_logits = True),
	metrics = ['accuracy'])

#evaluate the model
initial_epochs = 3
validation_steps = 20
loss0,accuracy0 = model.evaluate(validation_batches, steps = validation_steps)

history = model.fit(train_batches, 
	epochs = initial_epochs,
	validation_data = validation_batches)

acc = history.history['accuracy']
print(acc)