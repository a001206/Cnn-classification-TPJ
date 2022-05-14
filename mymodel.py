import pathlib
import PIL
import PIL.Image
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
import glob
#os.path.join 경로 이어줌 
#%%
# Path = '/Users/ganghaeseong/Documents/tf/imgs'
# train_dir = os.path.join(Path, 'train')
# validation_dir = os.path.join(Path, 'validation')

# BATCH_SIZE = 32
# IMG_SIZE = (160,160)

# train_dataset = image_dataset_from_directory(train_dir, shuffle=True, batch_size = BATCH_SIZE, image_size = IMG_SIZE)

data_dir = pathlib.Path('O_or_R')
train_dir = os.path.join(data_dir, 'TRAIN')
train_O_dir = os.path.join(train_dir, 'O')
train_R_dir = os.path.join(train_dir, 'R')

val_dir = os.path.join(data_dir, 'TEST')
val_O_dir = os.path.join(val_dir, 'O')
val_R_dir = os.path.join(val_dir, 'R')

# image_count = len(list(data_dir.glob('*/*/*.jpg')))
# print(image_count)

# #glob으로 리스트 만들고 시각화
# O = list(data_dir.glob('train/O/*.jpg'))
# img = PIL.Image.open(str(O[10]))
# # img.show()

batch_size = 32
img_height = 180
img_width = 180

train_ds = ImageDataGenerator(rescale = 1./255)
val_ds = ImageDataGenerator(rescale = 1./255)

train_ds = image_dataset_from_directory(train_dir,
                                        shuffle = True,
                                        batch_size =batch_size,
                                        image_size =(img_height, img_width))
val_ds = image_dataset_from_directory(val_dir,
                                      shuffle = True,
                                      batch_size = batch_size,
                                      image_size = (img_height, img_width))


# 시각화하기
# plt.figure(figsize=(10, 10))
# for images, labels in train_ds.take(1):
#   for i in range(9):
#     ax = plt.subplot(3, 3, i + 1)
#     plt.imshow(images[i].numpy().astype("uint8"))
#     plt.title(class_names[labels[i]])
#     plt.axis("off")
# plt.show()




num_classes = 2

model = Sequential([
    layers.experimental.preprocessing.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
    layers.Conv2D(64, 3, activation = 'relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes)
])

model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])


epochs=5
history = model.fit(train_ds,  
          validation_data=val_ds,
          epochs=epochs,) 

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss=history.history['loss']
val_loss=history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.ylim(0.6,1)
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.ylim(0,0.5)
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

model.save('55model/55model.h5')