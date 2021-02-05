#recursive neural network to generate a play

from keras.preprocessing import sequence
import keras
import tensorflow as tf 
import os
import numpy as np 

#load dataset
path_to_file = tf.keras.utils.get_file('shakespeare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')

# #to use own data, use this code--------------------------------------
# from google.colab import files
# path_to_file = list(files.upload().keys())[0]
#---------------------------------------------------------------------------------------

#read contents of file
text = open(path_to_file, 'rb').read().decode(encoding = 'utf-8')#read and decode for py2 compat
print('Length of text: {} characters'.format(len(text)))#length of text is num of char
print(text[:250])

#need to encode the text------------------------------------
vocab = sorted(set(text))
#creating a mapping from unique charactes to indices
char2idx = {u:i for i, u in enumerate(vocab)}
idx2char = np.array(vocab)

def text_to_int(text):
	return np.array([char2idx[c] for c in text])

text_as_int = text_to_int(text)
#--------------------------------------------------------------

# print("Text: ", text[:13])
# print("Encoded: ", text_to_int(text[:13]))

#create function to go from int to text
def int_to_text(ints):
	try:
		ints = ints.numpy()
	except:
		pass
	return ''.join(idx2char[ints])

# print(int_to_text(text_as_int[:13]))

#create training examples
seq_length = 100 # length of sequence for a training example
example_per_epoch = len(text)//(seq_length + 1)

#create training examples/targets
char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)
#use batch method to turn stream of characters into batches of desired length
sequences = char_dataset.batch(seq_length + 1, drop_remainder = True)
#use sequences of length 101 and split into input and output
def split_input_target(chunk):
	input_text = chunk[:-1]
	target_text = chunk[1:]
	return input_text,target_text

#use map to apply above function to every entry
dataset = sequences.map(split_input_target)

for x,y in dataset.take(2):
	print("\n\nEXAMPLE\n")
	print("INPUT")
	print(int_to_text(x))
	print("\nOUTPUT")
	print(int_to_text(y))