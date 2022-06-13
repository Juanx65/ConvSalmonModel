import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential


def ConvSalmonModel():
    class_names = ['salmon1','salmon2','salmon3','salmon5','salmon7','salmon9','salmon10','salmon11']#['salmon1','salmon2','salmon3','salmon4','salmon5','salmon6','salmon7','salmon8','salmon9','salmon10','salmon11']
    class_names_label = {class_name:i for i, class_name in enumerate(class_names)}
    nb_classes  = len(class_names)
    #print(class_name_label)
    IMAGE_SIZE = (200,100)

    model = tf.keras.Sequential([
        layers.experimental.preprocessing.Rescaling(1./255, input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)),
        layers.Conv2D(16,(3, 3), activation = 'relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, (3,3), activation = 'relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation = 'relu'),
        layers.Dense(nb_classes)
    ])

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    model.summary()
    return model

def main():
	model = ConvSalmonModel()

if __name__ == '__main__':
	main()
