import tensorflow as tf
import IPython.display as display

import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.figsize'] = (12,12)
mpl.rcParams['axes.grid'] = False

import numpy as np
import PIL.Image
import time
import functools

def tensor_to_image(tensor):
  tensor = tensor*255
  tensor = np.array(tensor, dtype=np.uint8)
  if np.ndim(tensor)>3:
    assert tensor.shape[0] == 1
    tensor = tensor[0]
  return PIL.Image.fromarray(tensor)

def load_img(path_to_img):
  max_dim = 512
  img = tf.io.read_file(path_to_img)
  img = tf.image.decode_image(img, channels=3)
  # print(img)
  img = tf.image.convert_image_dtype(img, tf.float32)

  shape = tf.cast(tf.shape(img)[:-1], tf.float32)
  long_dim = max(shape)
  scale = max_dim / long_dim

  new_shape = tf.cast(shape * scale, tf.int32)
  # new_shape = tf.cast()
  

  img = tf.image.resize(img, new_shape)
  img = img[tf.newaxis, :]
  return img

def imshow(image, title=None):
  if len(image.shape) > 3:
    image = tf.squeeze(image, axis=0)

  plt.imshow(image)
  if title:
    plt.title(title)

content_path = 'YellowLabradorLooking_new.jpg'
style_path = 'Vassily_Kandinsky,_1913_-_Composition_7.jpg'
import cv2
i1 = cv2.imread(content_path)
i2 = cv2.imread(style_path)

w1,h1,c1 = i1.shape
w2,h2,c2 = i2.shape

if w1+h1 > w2+h2:
  new_width = w1
  new_height = h1
else:
  new_width = w2
  new_height = h2

content_image = load_img(content_path)
style_image = load_img(style_path)
# print(tf.constant(content_image).numpy())


import tensorflow_hub as hub
import cv2
hub_module = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/1')
stylized_image = hub_module(tf.constant(style_image), tf.constant(content_image))[0]
stylized_image = tf.image.resize(stylized_image, (new_width,new_height))
# print(stylized_image)
out = tf.constant(stylized_image).numpy()
out = out.reshape(new_width, new_height, 3)
print(out.shape)
# # image = tensor_to_image(stylized_image)
# cv2.imwrite('out.jpg',out)
plt.imsave('out.jpg',out)
# cv2.imshow('oy',out)
# cv2.waitKey()
# image.save('out.jpg')
# img2 = cv2.imread('Harsh.png')
# data_tf = tf.convert_to_tensor(img2, np.float32)
# print(img2.dtype)
# print(img2.shape)