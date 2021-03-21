from tensorflow.keras.applications import Xception
from tensorflow.keras import Model
from tensorflow.keras import Sequential
from tensorflow.keras import Input
from tensorflow.keras.layers import *
from tensorflow import expand_dims
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf

from skimage import io
from skimage.transform import resize
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

print("VGG16:")
print("input shape: "+str(input_shape)) 
print("extractor layer: name={}, index={}".format(extractor_layer["name"], extractor_layer["index"]))

extractor_model = Model(inputs=pretrained_model.input,
                        outputs=pretrained_model.get_layer(index=extractor_layer["index"]).output)


print("extractor model:")
print("input shape: "+str(extractor_model.input_shape))
print("output shape: "+str(extractor_model.output_shape))

classifier_model = Sequential([
                              extractor_model,
                              Flatten(),
                              Dense(1, activation="sigmoid", kernel_regularizer=tf.keras.regularizers.l1_l2(l1=1e-5, l2=1e-4),)
])

classifier_model.compile(optimizer="nadam", metrics=["accuracy"], loss=tf.keras.losses.BinaryCrossentropy(from_logits=True))
classifier_model.summary()

def a(img):
  orwidth,orheight,ornum_channel=img.shape
  img = skimage.transform.resize(img, (224,224,3))#resize
  height,width,num_channel=img.shape#get image's width height and channel(color mode)
  minlength = max((width, height))

  ranbool = random.randint(0,2)
  if ranbool==1:csize=int(minlength*1/2)#define cropping size
  if ranbool==0:csize=int(minlength*3/4)#define crop size
  if ranbool==2:csize=int(minlength*4/4)#define crop size
  x=random.randint(0,img.shape[1]-csize)#define x localization of cropping frame
  y=random.randint(0,img.shape[0]-csize)#define y localization of cropping frame
  img = img[x:x+csize, y:y+csize]#crop

  #if random.randint(0,3)==2:img = 255-img
  if random.randint(0,9)==1:img = skimage.util.random_noise(img, clip=False, mode="gaussian")

  return skimage.transform.resize(img, (224,224,3))

augmentation = tf.keras.preprocessing.image.ImageDataGenerator(rotation_range=360,
                                                          fill_mode="reflect", 
                                                          #width_shift_range=0.22, height_shift_range=0.22,
                                                          shear_range=0.07,
                                                          zoom_range=0.2,
                                                          brightness_range=(0.3, 0.8),
                                                          horizontal_flip=True, vertical_flip=True,
                                                          preprocessing_function=a)


import tensorflow_datasets as tfds
tfds.load("div2k", split=["train"])

import cv2 as cv
normalize = tf.keras.Sequential([
                                tf.keras.layers.experimental.preprocessing.Resizing(224,224),
                                tf.keras.layers.experimental.preprocessing.Rescaling(1./255),
])

batch_size = 32

posimg = cv.imread("/content/img.jpg")
posimg = normalize(posimg)

negdir = r"/root/tensorflow_datasets/downloads/extracted/ZIP.data.visi.ee.ethz.ch_cvl_DIV2_DIV2_trai_LR0TdBvQgijDd9djBMXTGmUKOJiAMMz6y9364yO6TSPfA.zip/DIV2K_train_LR_bicubic"
neg = tf.keras.preprocessing.image_dataset_from_directory(negdir, image_size=(224,224), batch_size=batch_size)
negx = neg.map(lambda x,y:normalize(x))

posx = []
for i in range(batch_size):
  posx.append(posimg)
posx = tf.data.Dataset.from_tensor_slices(posx).batch(batch_size)

for i in posx:
  posx = i

posy = []
negy = []
for i in range(batch_size):
  negy.append(0)
  posy.append(1)


num_epochs = 5
epoch = 0
history = []

for negxbatch in negx:
    epoch+=1
    print("epoch:"+str(epoch))
    step = classifier_model.fit(augmentation.flow(np.concatenate((posx, negxbatch), axis=0),
                                np.concatenate((posy, negy), axis=0),
                                batch_size=32), validation_data = (augmentation.flow(np.concatenate((posx, negxbatch), axis=0),
                                np.concatenate((posy, negy), axis=0),
                                batch_size=8)),
                                epochs=1)

    history.append(step)

    if epoch==num_epochs:break
      
loss = []
acc = []
vloss = []
vacc = []
import matplotlib.pyplot as plt
for step in history:
  loss.append(step.history["loss"])
  acc.append(step.history["accuracy"])
  vloss.append(step.history["val_loss"])
  vacc.append(step.history["val_accuracy"])

plt.plot(range(5), acc)
plt.plot(range(5), vacc)

from PIL import Image

dir = "/content/img123.jpg"
img = Image.open(dir).convert('RGB')
img = tf.expand_dims(np.array(np.asarray(img.resize((224,224)))), 0)
print(float(classifier_model.predict(img)))
plt.imshow(tf.squeeze(img))
