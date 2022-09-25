from tensorflow.keras.applications import Xception, VGG19
from tensorflow.keras import Model, Sequential, layers
import tensorflow as tf

import cv2 as cv

pretrained_model = Xception()
pretrained_model.trainable = False

input_shape = pretrained_model.input_shape

ind = -2
extractor_layer = {"name": pretrained_model.layers[ind].name, "index": ind}

print("Xception:")
print("input shape: "+str(input_shape)) 
print(f"extractor layer: name={extractor_layer['name']}, index={extractor_layer['index']}")
print('\n')

extractor_model = Model(
  inputs=pretrained_model.input,
  outputs=pretrained_model.get_layer(index=extractor_layer["index"]).output
)


print("extractor model:")
print("input shape: "+str(extractor_model.input_shape))
print("output shape: "+str(extractor_model.output_shape))
print('\n')

normalize = Sequential([
  layers.experimental.preprocessing.Resizing(299,299), 
  layers.experimental.preprocessing.Rescaling(1./255),
])

positive_img_path = input('enter path for positive image: \n > ')
negative_img_path = input('enter path for newgative image: \n > ')

pos_img = cv.imread(positive_img_path)

pos_img = tf.expand_dims(normalize(pos_img), 0)
pos_feature_map = extractor_model.predict(pos_img)[0]

neg_img = cv.imread(negative_img_path)

neg_img = tf.expand_dims(normalize(neg_img), 0)
neg_feature_map = extractor_model.predict(neg_img)[0]

print(abs(pos_feature_map-neg_feature_map).sum()/2048)
