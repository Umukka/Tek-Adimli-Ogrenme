from tensorflow.keras.applications import Xception, VGG19
from tensorflow.keras import Model, Input, Sequential
from tensorflow.keras.layers import *

from tensorflow import expand_dims
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf

import numpy as np
import cv2 as cv
import skimage
import random

pretrained_model = Xception()
pretrained_model.trainable = False

input_shape = pretrained_model.input_shape

layers = [] 
for layer in pretrained_model.layers:
  layers.append(layer)
ind = -2
extractor_layer = {"name": layers[ind].name, "index": ind}

print("Xception:")
print("input shape: "+str(input_shape)) 
print(f"extractor layer: name={extractor_layer["name"]}, index={extractor_layer["index"]}")

"""
pretrained_model = VGG19()
pretrained_model.trainable = False

input_shape = pretrained_model.input_shape

layers = [] 
for layer in pretrained_model.layers:
  layers.append(layer)
ind = -3
extractor_layer = {"name":layers[ind].name, "index":ind}

print("VGG16:")
print("input shape: "+str(input_shape)) 
print("extractor layer: name={}, index={}".format(extractor_layer["name"], extractor_layer["index"]))
"""

extractor_model = Model(inputs=pretrained_model.input,
                        outputs=pretrained_model.get_layer(index=extractor_layer["index"]).output)


print("extractor model:")
print("input shape: "+str(extractor_model.input_shape))
print("output shape: "+str(extractor_model.output_shape))

normalize = tf.keras.Sequential([
                                tf.keras.layers.experimental.preprocessing.Resizing(299,299),
                                tf.keras.layers.experimental.preprocessing.Rescaling(1./255),
])

pos_img = cv.imread("/content/img1.jpg")

pos_img = tf.expand_dims(normalize(pos_img), 0)
posf = extractor_model.predict(posimg)[0]

neg_img = cv.imread("/content/img2.jpg")

neg_img = tf.expand_dims(normalize(neg_img), 0)
negf = extractor_model.predict(neg_img)[0]
abs(posf-negf).sum()/2048
