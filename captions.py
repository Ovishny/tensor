#image captioning on MS-COCO dataset

import tensorflow as tf 

#for charting plots
import matplotlib.pyplot as plt 

#scikit-learn has helpful utilities
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

import collections
import random
import re
import numpy as np 
import os
import time
import json
from glob import glob
from PIL import Image 
import pickle

#download caption annotation files
annotation_folder = '/annotations/'


if not os.path.exists(os.path.abspath('.') + annotation_folder):
	annotation_zip = tf.keras.utils.get_file('captions.zip',
		cache_subdir = os.path.abspath('.'),
		origin = 'http://images.cocodataset.org/annotations/annotations_trainval2014.zip',
		extract = True)
	annotation_file = os.path.dirname(annotation_zip) + '\\annotations\\captions_train2014.json'
	os.remove(annotation_zip)
else:
	annotation_file = os.path.abspath('.') + '/annotations/captions_train2014.json'

print(annotation_file)
#download image files
image_folder = '/train2014/'
if not os.path.exists(os.path.abspath('.') + image_folder):
	image_zip = tf.keras.utils.get_file('train2014.zip',
		cache_subdir = os.path.abspath('.'),
		origin = 'http://images.cocodataset.org/zips/train2014.zip',
		extract = True)
	PATH = os.path.dirname(image_zip) + image_folder
	os.remove(image_zip)
else:
	PATH = os.path.abspath('.') + image_folder

print(PATH)

#optional: limit size of training set. Gonna use 30000, using more would result in improved captioning quality
with open(annotation_file, 'r') as f:
	annotations = json.load(f)
#group captions together with same id
image_path_to_caption = collections.defaultdict(list)
for val in annotations['annotations']:
	caption = f"<start> {val['caption']} <end>"
	image_path = PATH + 'COCO_train2014_' + '%012d.jpg' % (val['image_id'])
	image_path_to_caption[image_path].append(caption)
image_paths = list(image_path_to_caption.keys())
random.shuffle(image_paths)
#select first 6000 image_paths, each id has 5 captions, leading to 30,000 examples
train_image_paths = image_paths[:6000]
train_captions = []
img_name_vector = []

for image_path in train_image_paths:
	caption_list = image_path_to_caption[image_path]
	train_captions.extend(caption_list)
	img_name_vector.extend([image_path] *len(caption_list))

# #look at an example
# print(train_captions[0])
# im = Image.open(img_name_vector[0])
# im.show()

#preprocess image using inceptionv3, pretrained on imagenet
#convert image to expected format, 299pxX299px, and preprocess image using preprocess_input to normalize image between values of -1 and 1
def load_image(image_path):
	img = tf.io.read_file(image_path)
	img = tf.image.decode_jpeg(img, channels = 3)
	img = tf.image.resize(img, (299, 299))
	img = tf.keras.applications.inception_v3.preprocess_input(img)
	return img, image_path

#initialize inceptionV3 and load pretrained imagenet weights
#create tf.keras model with output layer last convolutional layer in inceptionv3 architecture
#forward each image through network and store resulting vector in dictionary
#after all images passed through, pickle the dictionary and save to disk
image_model = tf.keras.applications.InceptionV3(include_top = False,
	weights = 'imagenet')
new_input = image_model.input
hidden_layer = image_model.layers[-1].output
image_features_extract_model = tf.keras.Model(new_input, hidden_layer)

#get unique images
encode_train = sorted(set(img_name_vector))

#can change batch_size according to system config
image_dataset= tf.data.Dataset.from_tensor_slices(encode_train)
image_dataset = image_dataset.map(
	load_image, num_parallel_calls = tf.data.AUTOTUNE).batch(16)

for img, path in image_dataset:
	batch_features = image_features_extract_model(img)
	batch_features = tf.reshape(batch_features, (batch_features.shape[0], -1, batch_features.shape[3]))

	for bf, p in zip(batch_features, path):
		path_of_feature = p.numpy().decode('utf-8')
		np.save(path_of_feature, bf.numpy())

#preprocess and tokenize captions
#tokenize captions splits on spaces giving vocab of unique words
#limit vocab size to top 5000 words to save memory, eberything else becomes "UNK"
#create word-to-index and index-to-word mapping
#pad all sequences to be same length as longest one

#find max length
def calc_max_length(tensor):
	return max(len(t) for t in tensor)

#choose top 5000 words from vocab
top_k = 5000
tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words = top_k,
	oov_token = '<unk>',
	filters ='!"#$%&()*+.,-/:;=?@[\]^_`{|}~ ' )

tokenizer.fit_on_texts(train_captions)
tokenizer.word_index['<pad>'] = 0
tokenizer.index_word[0] = '<pad>'

#create tokenized vectors
train_seqs = tokenizer.texts_to_sequences(train_captions)

#pad each vector to max_length of captions. If not provided max length, pad sequences will calculate it automatically
cap_vector = tf.keras.preprocessing.sequence.pad_sequences(train_seqs, padding = 'post')

#calculate max_length, store the attention weights
max_length = calc_max_length(train_seqs)

#split data into training and testing
img_to_cap_vector = collections.defaultdict(list)
for img, cap in zip(img_name_vector, cap_vector):
	img_to_cap_vector[img].append(cap)

#create training and validation sets using an 80-20 split randomly
img_keys = list(img_to_cap_vector.keys())
random.shuffle(img_keys)

slice_index = int(len(img_keys)*0.8)
img_name_train_keys, img_name_val_keys = img_keys[:slice_index], img_keys[slice_index:]

img_name_train = []
cap_train = []
for imgt in img_name_train_keys:
	capt_len = len(img_to_cap_vector[imgt])
	img_name_train.extend([imgt]*capt_len)
	cap_train.extend(img_to_cap_vector[imgt])

img_name_val = []
cap_val = []
for imgv in img_name_val_keys:
	capv_len = len(img_to_cap_vector[imgv])
	img_name_val.extend([imgv]*capv_len)
	cap_val.extend(img_to_cap_vector[imgv])

# print(len(img_name_train), len(cap_train), len(img_name_val), len(cap_val))

#create tf.data dataset for training
#Can change parameters based on system config
BATCH_SIZE = 64
BUFFER_SIZE = 1000
embedding_dim = 256
units = 512
vocab_size = top_k + 1
num_steps = len(img_name_train)//BATCH_SIZE
#shape vector extracted is (64,2048) these two variables represent that
feature_shape = 2048
attention_features_shape = 64

#load numpy files
def map_func(img_name, cap):
	img_tensor = np.load(img_name.decode('utf-8') + '.npy')
	return img_tensor, cap

dataset = tf.data.Dataset.from_tensor_slices((img_name_train, cap_train))

#use map to load numpy files in parallel
dataset = dataset.map(lambda item1, item2: tf.numpy_function(
	map_func, [item1, item2], [tf.float32, tf.int32]),
	num_parallel_calls = tf.data.AUTOTUNE)

#shuffle and batch
dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
dataset = dataset.prefetch(buffer_size = tf.data.AUTOTUNE)

#create model
#extract features from lower convolutional layer, giving vector of (8,8,2048)
#squash to shape (64,2048), then passed through cnn encoder
#the rnn attends over image to predict the next word
class BahdanauAttention(tf.keras.Model):
	def __init__(self, units):
		super(BahdanauAttention, self).__init__()
		self.W1 = tf.keras.layers.Dense(units)
		self.W2 - tf.keras.layers.Dense(units)
		self.V = tf.keras.layers.Dense(1)

	def call(self, features, hidden):
		#features cnn encoder output shape == batch size, 64, embedding_dim
		#hidden shape == batch_size, hidden_size
		#hidden with time axis shape == batch_size, 1, hidden_size
		hidden_with_time_axis = tf.expand_dims(hidden, 1)

		#attention_hidden_layer shape = batch_size, 64 ,units
		attention_hidden_layer = (tf.nn.tanh(self.W1(features)+
			self.W2(hidden_with_time_axis)))

		#score shape == batch_size, 64, 1
		#this gives unnormalized score for each image feature
		score = self.V(attention_hidden_layer)

		#attention_weights shape = batch_size, 64, 1
		attention_weights = tf.nn.softmax(score, axis = 1)

		#context vector shape after sum == batch size, hidden size
		context_vector = attention_weights*features
		context_vector = tf.reduce_sum(context_vector, axis = 1)

		return context_vector, attention_weights

class CNN_Encoder(tf.keras.Model):
	#since extracted features already and dumped using pickle
	#encoder passes feature through fully connected layer
	def __init__(self, embedding_dim):
		super(CNN_Encoder, self).__init__()
		#shape after fc == batch_size, 64, embedding_dim
		self.fc = tf.keras.layers.Dense(embedding_dim)

	def call(self, x):
		x = self.fc(x)
		x = tf.nn.relu(x)
		return x 

class RNN_Decoder(tf.keras.Model):

	def __init__(self, embedding_dim, units, vocab_size):
		super(RNN_Decoder, self).__init__()
		self.units = units

		self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
		self.gru = tf.keras.layers.GRU(self.units, return_sequences = True, return_state = True, recurrent_initializer = 'glorot_uniform')
		self.fc1 = tf.keras.layers.Dense(self.units)
		self.fc2 = tf.keras.layers.Dense(vocab_size)

		self.attention = BahdanauAttention(self.units)

	def call(self, x, features, hidden):
		#define attention as model
		context_vector, attention_weights = self.attention(features, hidden)

		#x shape after passing through embedding == batch size, 1, embedding_Dim
		x = self.embedding(x)

		#x shape after concatenation == batch_size, 1, embedding_dim + hidden_size
		x = tf.concat([tf.expand_dims(context_vector, 1), x], axis = -1)

		#pass concatenated vector to gru
		output, state = self.gru(x)

		#shape == batch_size, max_length, hidden_size
		x = self.fc1(output)

		#x shape == batch_size*max_length, hidden_size
		x = tf.reshape(x, (-1,x.shape[2]))

		#output shape == batch_size*max_length, vocab
		x = self.fc2(x)

		return x, state, attention_weights

	def reset_state(self, batch_size):
		return tf.zeros((batch_Size, self.units))

encoder = CNN_Encoder(embedding_dim)
decoder = RNN_Decoder(embedding_dim, units, vocab_size)

optimizer = tf.keras.optimizers.Adam()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
	from_logits = True, reductions = 'none')

def loss_function(real, pred):
	mask = tf.math.logical_not(tf.math.equal(real, 0))
	loss_ = loss_object(real, pred)

	mask = tf.cast(mask, dtype= loss_.dtype)
	loss_ *= mask

	return tf.reduce_mean(loss_)