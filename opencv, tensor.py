from importlib.resources import path
import pathlib
import PIL
import PIL.Image
from cv2 import imread
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import glob
from glob import glob
import cv2

cam = cv2.VideoCapture(0)
img_height = 180
img_width = 180

model = tf.keras.models.load_model('my_model/O_R_model.h5')

while True:
    ret, frame = cam.read()
    frame = cv2.flip(frame, 1)
    if not ret:
        print("failed to grab frame")
        break
    cv2.imshow("opencv test", frame)

    k = cv2.waitKey(1)
    if k%256 == 27:
        # ESC pressed               
        print("Escape hit, closing...")
        break
    elif k%256 == 32:
        # SPACE pressed
        img_name = "img.png"
        cv2.imwrite(img_name, frame)
        print('good')
        cv2.imshow('img.png', img_name)
        # pre_img = imread('img.png', rgb_image)

        # # pre_img = '/Users/ganghaeseong/Desktop/다운로드.jpeg'

        # img = keras.preprocessing.image.load_img(
        # pre_img, target_size=(img_height, img_width)
        # )
        # img_array = keras.preprocessing.image.img_to_array(img)
        # img_array = tf.expand_dims(img_array, 0) #Create a batch

        # predictions = model.predict(img_array)
        # predictions = np.argmax(predictions)