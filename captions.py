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