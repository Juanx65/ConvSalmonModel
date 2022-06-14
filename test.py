import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

import argparse

#from tflite_model_maker import image_classifier
#from tflite_model_maker import ImageClassifierDataLoader
#from tflite_model_maker.image_classifier import DataLoader

from model import ConvSalmonModel

def predict_class_label_number(dataset):
  """Runs inference and returns predictions as class label numbers."""
  rev_label_names = {l: i for i, l in enumerate(label_names)}
  return [
      rev_label_names[o[0][0]]
      for o in model.predict_top_k(dataset, batch_size=128)
  ]

def show_confusion_matrix(cm, labels):
  plt.figure(figsize=(10, 8))
  sns.heatmap(cm, xticklabels=labels, yticklabels=labels,
              annot=True, fmt='g')
  plt.xlabel('Prediction')
  plt.ylabel('Label')
  plt.show()


label_names = ['salmon1','salmon2','salmon3','salmon4','salmon5','salmon6','salmon7','salmon8','salmon9','salmon10','salmon11']#
IMAGE_SIZE = (200,100)
#Entrenar el modelo
batch_size = 32#opt.batch_size
epochs = 100#opt.epochs
data_dir = "rois/"#opt.data_dir #"rois/"

ds_test = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  labels='inferred',
  label_mode='int',
  class_names = label_names,
  #validation_split=0.2,
  #subset="test",
  seed=123,
  image_size=(IMAGE_SIZE[0], IMAGE_SIZE[1]),
  batch_size=batch_size)
#test_data = ImageClassifierDataLoader(ds_test, ds_test.cardinality(),label_names)
#test_data = DataLoader.from_folder(data_dir)

model = ConvSalmonModel('adam', 0.8)

confusion_mtx = tf.math.confusion_matrix(
    list(ds_test.map(lambda x, y: y)),
    predict_class_label_number(ds_test),
    num_classes=len(label_names))

show_confusion_matrix(confusion_mtx, label_names)
