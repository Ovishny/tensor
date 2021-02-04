#Natural Language Processing on Movie Revies using recurrent neural network

from keras.datasets import imdb
from keras.preprocessing import sequence
import tensorflow as tf 
import os 
import numpy as np 

VOCAB_SIZE = 88584

MAXLEN = 250
BATCH_SIZE = 64

#split data from dataset
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words = VOCAB_SIZE)

#padding reviews so each review has same length when passed in
train_data = sequence.pad_sequences(train_data, MAXLEN)
test_data = sequence.pad_sequences(test_data, MAXLEN)

#creating the model
model = tf.keras.Sequential([
	tf.keras.layers.Embedding(VOCAB_SIZE,32), #word embedding layer
	tf.keras.layers.LSTM(32), #LSTM layers
	tf.keras.layers.Dense(1, activation = 'sigmoid') #Dense layer. Sigmoid squishes values between 0 and 1. <.5 is negative and >.5 is positive
])

#train the model
model.compile(loss = "binary_crossentropy",
	optimize = 'rmsprop',
	metrics = ['acc']
)
history = model.fit(train_Data, train_labels, epochs = 10, validation_split = 0.2)