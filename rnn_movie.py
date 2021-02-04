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
	optimizer = 'rmsprop',
	metrics = ['acc']
)
history = model.fit(train_data, train_labels, epochs = 10, validation_split = 0.2)

#evaluate the model
results = model.evaluate(test_data, test_labels)
print(results)

#make an encode function
word_index = imdb.get_word_index() #get index of imdb dataset

def encode_text(text):
	tokens = tf.keras.preprocessing.text.text_to_word_sequence(text) #convert text to tokens
	tokens = [word_index[word] if word in word_index else 0 for word in tokens] #if word is in mapping replace location in list with corresponding index, else 0
	return sequence.pad_sequences([tokens], MAXLEN)[0] #pad it, but only takes list of sequences so we put tokens in a list so we only need the 0 index to get the token sequence

text = 'that movie was just amazing, so amazing'
encoded = encode_text(text)
print(encoded)

#make a decode function
reverse_word_index = {value: key for (key,value) in word_index.items()}

def decode_integers(integers):
	PAD = 0
	text = ''
	for num in integers:
		if num != PAD:
			text += reverse_word_index[num] + ' '
	return text[:-1]

print(decode_integers(encoded))

#make a prediction

def predict(text):
	encoded_text = encode_text(text)
	pred = np.zeros((1,250))
	pred[0] = encoded_text
	result = model.predict(pred)
	print(result[0])

positive_review = "That movie was so awesome! I really loved it and would wait it again because it was amazingly great"
predict(positive_review)

negative_review = "that movie sucked. I hated it and wouldn't watch it again. Was one of the worst things I've ever watched"
predict(negative_review) 