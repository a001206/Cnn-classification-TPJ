import pathlib
import PIL
import PIL.Image
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
from keras.preprocessing import image_dataset_from_directory
from tensorflow import keras
from keras import layers
from keras.models import Sequential
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
train_iu_dir = os.path.join(train_dir, 'O')
train_kwon_dir = os.path.join(train_dir, 'R')

val_dir = os.path.join(data_dir, 'TEST')
val_iu_dir = os.path.join(val_dir, 'O')
val_kwon_dir = os.path.join(val_dir, 'R')

image_count = len(list(data_dir.glob('*/*/*.jpg')))
print(image_count)

#glob으로 리스트 만들고 시각화
O = list(data_dir.glob('train/O/*.jpg'))
img = PIL.Image.open(str(O[10]))
# img.show()

batch_size = 32
img_height = 180
img_width = 180

train_ds = image_dataset_from_directory(train_dir,
                                        shuffle = True,
                                        batch_size =batch_size,
                                        image_size =(img_height, img_width))
val_ds = image_dataset_from_directory(val_dir,
                                      shuffle = True,
                                      batch_size = batch_size,
                                      image_size = (img_height, img_width))

# #훈련세트 비율 나누기
# train_ds = tf.keras.preprocessing.image_dataset_from_directory(
#   data_dir,
#   validation_split=0.2,
#   subset="training",
#   seed=123,
#   image_size=(img_height, img_width),
#   batch_size=batch_size)
# #검증세트 "
# val_ds = tf.keras.preprocessing.image_dataset_from_directory(
#   data_dir,
#   validation_split=0.2,
#   subset="validation",
#   seed=123,
#   image_size=(img_height, img_width),
#   batch_size=batch_size)

class_names = train_ds.class_names
# print(class_names)


# 시각화하기
# plt.figure(figsize=(10, 10))
# for images, labels in train_ds.take(1):
#   for i in range(9):
#     ax = plt.subplot(3, 3, i + 1)
#     plt.imshow(images[i].numpy().astype("uint8"))
#     plt.title(class_names[labels[i]])
#     plt.axis("off")
# plt.show()

#
# for image_batch, labels_batch in train_ds:
#     print(image_batch.shape)
#     print(labels_batch.shape)
#     break

AUTOTUNE = tf.data.experimental.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1./255)

normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(iter(normalized_ds))
first_image = image_batch[0]
# Notice the pixels values are now in `[0,1]`.
# print(np.min(first_image), np.max(first_image)

num_classes = 2
#model
#%%
model = Sequential([
    layers.experimental.preprocessing.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes)
    ])

model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

# history = model.fit(                     #나중에 정확도 출렷할 때 history사용
#   train_ds,
#   validation_data=val_ds,           
#   epochs=epochs
# )

epochs=10
history = model.fit(train_ds,  
          validation_data=val_ds,
          epochs=epochs,) 


# model.summary()

#훈련한 모델 저장
# model.fit(train_ds, epochs = 5)
# model.save('save_model')

# new_model = tf.keras.models.load_model('save_model')
# new_model.summary()



# checkpoint_path = "training_1/cp.ckpt"
# checkpoint_dir = os.path.dirname(checkpoint_path)

# # 모델의 가중치를 저장하는 콜백 만들기
# cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
#                                                  save_weights_only=True,
#                                                  verbose=1)

# # 새로운 콜백으로 모델 훈련하기
# model.fit(train_ds,   
#           epochs=10,
#           validation_data=val_ds,
#           callbacks=[cp_callback])  # 콜백을 훈련에 전달합니다
          



# 옵티마이저의 상태를 저장하는 것과 관련되어 경고가 발생할 수 있습니다.
# 이 경고는 (그리고 이 노트북의 다른 비슷한 경고는) 이전 사용 방식을 권장하지 않기 위함이며 무시해도 좋습니다.


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss=history.history['loss']
val_loss=history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

# with tf.device('/cpu:0'): #개빡쳐 M1아직 증강 지원 안 함;; 이거 붙여줘야해
#   data_augmentation = keras.Sequential(
#     [
#       layers.experimental.preprocessing.RandomFlip("horizontal", 
#                                                   input_shape=(img_height, 
#                                                                 img_width,
#                                                                 3)),
#       layers.experimental.preprocessing.RandomRotation(0.1),
#       layers.experimental.preprocessing.RandomZoom(0.1),
#     ]
#   )

# # # plt.figure(figsize=(10, 10))
# # # for images, _ in train_ds.take(1):
# # #   for i in range(9):
# # #     augmented_images = data_augmentation(images)
# # #     ax = plt.subplot(3, 3, i + 1)
# # #     plt.imshow(augmented_images[0].numpy().astype("uint8"))
# # #     plt.axis("off")
# # # plt.show()

# model = Sequential([
#     data_augmentation,
#     layers.experimental.preprocessing.Rescaling(1./255),
#     layers.Conv2D(16, 3, padding='same', activation='relu'),
#     layers.MaxPooling2D(),
#     layers.Conv2D(32, 3, padding='same', activation='relu'),
#     layers.MaxPooling2D(),
#     layers.Conv2D(64, 3, padding='same', activation='relu'),
#     layers.MaxPooling2D(),
#     layers.Dropout(0.2),
#     layers.Flatten(),
#     layers.Dense(128, activation='relu'),
#     layers.Dense(num_classes)
# ])


# model.compile(optimizer='adam',
#                 loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#                 metrics=['accuracy'])

# # model.summary()

# epochs=10
# history = model.fit(
#   train_ds,
#   validation_data=val_ds,
#   epochs=epochs
#   ) 
# # epochs = 15
# # history = model.fit(train_ds,   
# #           validation_data=val_ds,
# #           epochs=epochs,)         #아까 저장한 콜백 다시 불러와서 model2학습

# # model.save('saved_model/my_model')
# # model.save('save_model/my_model.h5')  

# acc = history.history['accuracy']
# val_acc = history.history['val_accuracy']

# loss=history.history['loss']
# val_loss=history.history['val_loss']

# epochs_range = range(epochs)

# plt.figure(figsize=(8, 8))
# plt.subplot(1, 2, 1)
# plt.plot(epochs_range, acc, label='Training Accuracy')
# plt.plot(epochs_range, val_acc, label='Validation Accuracy')
# plt.legend(loc='lower right')
# plt.title('Training and Validation Accuracy')

# plt.subplot(1, 2, 2)
# plt.plot(epochs_range, loss, label='Training Loss')
# plt.plot(epochs_range, val_loss, label='Validation Loss')
# plt.legend(loc='upper right')
# plt.title('Training and Validation Loss')
# plt.show()


# #위에서 지정한 img_height = 180, img_width = 180  
# pre_img = '/Users/ganghaeseong/Desktop/다운로드 (1).jpeg'

# img = keras.preprocessing.image.load_img(
#     pre_img, target_size=(img_height, img_width)
# )
# img_array = keras.preprocessing.image.img_to_array(img)
# img_array = tf.expand_dims(img_array, 0) # Create a batch

# predictions = model.predict(img_array)
# score = tf.nn.softmax(predictions[0])

# print(
#     "This image most likely belongs to {} with a {:.2f} percent confidence."
#     .format(class_names[np.argmax(score)], 100 * np.max(score))
# )
model.save('my_model/O_R_model.h5')
# %%
