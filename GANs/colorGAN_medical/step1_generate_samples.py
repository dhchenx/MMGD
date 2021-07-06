#!/usr/bin/python
# coding:utf8
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
datagen = ImageDataGenerator(rotation_range=40,
                             width_shift_range=0.2,
                             height_shift_range=0.2,
                             shear_range=0.2,
                             zoom_range=0.2,
                             horizontal_flip=True,
                             fill_mode='nearest')



img = load_img('images/hand256.jpg')
x = img_to_array(img)
x = x.reshape((1,)+x.shape)

# hand
label="3"
i = 0
for batch in datagen.flow(x, batch_size=200,
                          save_to_dir='dataset/train/'+label, save_prefix='hand', save_format='jpg'):
    i += 1
    if i>12*1000:
        break

'''
img = load_img('images/heart256.jpg')
x = img_to_array(img)
x = x.reshape((1,)+x.shape)

# cxr
label="1"
i=0
for batch in datagen.flow(x, batch_size=200,
                          save_to_dir='dataset/train/'+label, save_prefix='heart', save_format='jpg'):
    i += 1
    if i>12*1000:
        break
'''
