import numpy as np
#import pandas as pd
import os
#from sklearn.metrics import classification_report
#import seaborn as sn; sn.set(font_scale=1.4)
#from sklearn.utils import shuffle
import matplotlib.pyplot as plt
#import cv2
import tensorflow as tf
#from tqdm import tqdm

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential


class_names = ['salmon1','salmon2','salmon3','salmon4','salmon5','salmon6','salmon7','salmon8','salmon9','salmon10','salmon11']
class_names_label = {class_name:i for i, class_name in enumerate(class_names)}

nb_classes  = len(class_names)
#print(class_name_label)
IMAGE_SIZE = (200,100)

model = tf.keras.Sequential([
    layers.Rescaling(1./255, input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)),
    layers.Conv2D(16,(3, 3), activation = 'relu'),
    layers.MaxPooling2D(),
    layer.Conv2D(32, (3,3), activation = 'relu'),
    layer.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation = 'relu'),
    layers.Dense(nb_classes)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.summary()

#Entrenar el modelo
batch_size = 32
epochs = 10
data_dir = "dir/to/dataset"

train_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(IMAGE_SIZE[0], IMAGE_SIZE[1]),
  batch_size=batch_size)

val_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(IMAGE_SIZE[0], IMAGE_SIZE[1]),
  batch_size=batch_size)

#para mantener el dataset cargado en la RAM y que sea más rapido
AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs
)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

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
