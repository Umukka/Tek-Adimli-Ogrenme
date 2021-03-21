import tensorflow as tf
import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from skimage import io#
import tensorflow_datasets as tfds

import skimage
import random
from skimage import filters
import sys
import cv2 as cv

tfds.load("div2k", split=["train"])

def a(img):
  orwidth,orheight,ornum_channel=img.shape
  img = skimage.transform.resize(img, (180,180,3))#resize
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

  return skimage.transform.resize(img, (orwidth,orheight,ornum_channel))

augmentation = tf.keras.preprocessing.image.ImageDataGenerator(rotation_range=360,
                                                          fill_mode="reflect", 
                                                          width_shift_range=0.22, height_shift_range=0.22,
                                                          shear_range=0.07,
                                                          zoom_range=0.2,
                                                          brightness_range=(0.3, 0.8),
                                                          horizontal_flip=True, vertical_flip=True,
                                                          preprocessing_function=a)

normalize = tf.keras.Sequential([
                                tf.keras.layers.experimental.preprocessing.Resizing(180,180),
                                tf.keras.layers.experimental.preprocessing.Rescaling(1./255),
])

batch_size = 64

posimg = io.imread("/content/dunya2.jpg")
posimg = normalize(posimg)

negdir = r"/root/tensorflow_datasets/downloads/extracted/ZIP.data.visi.ee.ethz.ch_cvl_DIV2_DIV2_trai_LR0TdBvQgijDd9djBMXTGmUKOJiAMMz6y9364yO6TSPfA.zip/DIV2K_train_LR_bicubic"
neg = tf.keras.preprocessing.image_dataset_from_directory(negdir, image_size=(180,180), batch_size=batch_size)
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

 model = tf.keras.models.Sequential([
  tf.keras.layers.experimental.preprocessing.Rescaling(1, input_shape=(180, 180, 3)),
  tf.keras.layers.Conv2D(64, 3, padding='same'),
  tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Conv2D(128, 3, padding='same'),
  tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Conv2D(256, 3, padding='same'),
  tf.keras.layers.Conv2D(256, 3, padding='same', activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(1, activation="sigmoid", kernel_regularizer=tf.keras.regularizers.l1_l2(l1=1e-5, l2=1e-4))
 ])

model.compile(optimizer="nadam", metrics=["accuracy"], loss=tf.keras.losses.BinaryCrossentropy(from_logits=True))

num_epochs = 20
epoch = 0
history = []

for negxbatch in negx:
    epoch+=1
    print("epoch:"+str(epoch))
    step = model.fit(augmentation.flow(np.concatenate((posx, negxbatch), axis=0),
                                np.concatenate((posy, negy), axis=0),
                                batch_size=batch_size), epochs=1)

    history.append(step)

    if epoch==num_epochs:break

loss = []
acc = []

for step in history:
  loss.append(step.history["loss"])
  acc.append(step.history["accuracy"])
"""
loss.pop(0)
loss.pop(0)
loss.pop(0)
"""
plt.plot(range(num_epochs), loss)
plt.plot(range(num_epochs), acc)

dir = "/veriarttirma/img.jpg
img = Image.open(dir)
img = tf.expand_dims(np.array(np.asarray(img.resize((180,180)))), 0)
print(float(model.predict(img)))
plt.imshow(tf.squeeze(img))
