import tensorflow as tf 
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt 

#load in dataset
fashion_mnist = keras.datasets.fashion_mnist 
#split data into testing and training
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

#class names
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

#preprocess to fit pixel values between 0 and 1
train_images = train_images/255.0
test_images = test_images/255.0

#building the model 
model = keras.Sequential([
	keras.layers.Flatten(input_shape = (28,28)),#input layer 
	keras.layers.Dense(128, activation = 'relu'),#hidden layer
	keras.layers.Dense(10, activation = 'softmax')#output layer
	])
#compile the neural network
model.compile(optimizer = 'adam',
	loss = 'sparse_categorical_crossentropy',
	metrics = ['accuracy'])

#fit the model
model.fit(train_images, train_labels, epochs = 10)

#check accuracy
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose = 1)
print('Test accuracy:', test_acc)


#script to verify predictions
COLOR = 'green'
plt.rcParams['text.color'] = COLOR
plt.rcParams['axes.labelcolor'] = COLOR

def predict(model, image, correct_label):
	class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
		'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
	prediction = model.predict(np.array([image]))
	predicted_class = class_names[np.argmax(prediction)]

	show_image(image, class_names[correct_label], predicted_class)

def show_image(img,label,guess):
	plt.figure()
	plt.imshow(img, cmap = plt.cm.binary)
	plt.title("Expected: " + label)
	plt.xlabel("Guess: " + guess)
	plt.colorbar()
	plt.grid(False)
	plt.show()

def get_number():
	while True:
		num = input("Pick a number: ")
		if num.isdigit():
			num = int(num)
			if 0 <= num <= 1000:
				return int(num)
		else:
			print("Try again...")

num = get_number()
img = test_images[num]
label = test_labels[num]
predict(model, img, label)
yesno = input("Another number?(y/n): ")
while yesno == 'y':
	num = get_number()
	img = test_images[num]
	label = test_labels[num]
	predict(model, img, label)
	yesno = input("Another number?(y/n): ")

