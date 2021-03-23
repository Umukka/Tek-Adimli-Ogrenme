from tensorflow.keras.applications import Xception
from tensorflow.keras.applications import VGG19
from tensorflow.keras import Model
from tensorflow.keras import Sequential
from tensorflow.keras import Input
from tensorflow.keras.layers import *
from tensorflow import expand_dims
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf

import skimage

import random

import numpy as np

pretrained_model = Xception()
pretrained_model.trainable = False

input_shape = pretrained_model.input_shape

layers = [] 
for layer in pretrained_model.layers:
  layers.append(layer)
ind = -2
extractor_layer = {"name":layers[ind].name, "index":ind}

print("Xception:")
print("input shape: "+str(input_shape)) 
print("extractor layer: name={}, index={}".format(extractor_layer["name"], extractor_layer["index"]))

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

import cv2 as cv
normalize = tf.keras.Sequential([
                                tf.keras.layers.experimental.preprocessing.Resizing(299,299),
                                tf.keras.layers.experimental.preprocessing.Rescaling(1./255),
])

posimg = cv.imread("/content/img1.jpg")

posimg = tf.expand_dims(normalize(posimg), 0)
posf = extractor_model.predict(posimg)[0]

negimg = cv.imread("/content/img2.jpg")

negimg = tf.expand_dims(normalize(negimg), 0)
negf = extractor_model.predict(negimg)[0]
abs(posf-negf).sum()/2048
