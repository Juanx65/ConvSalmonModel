import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential


def ConvSalmonModel(optimizer,dropout):
    class_names = ['salmon1','salmon2','salmon3','salmon4','salmon5','salmon6','salmon7','salmon8','salmon9','salmon10','salmon11']#['salmon1','salmon2','salmon3','salmon5','salmon7','salmon9','salmon10','salmon11']#['salmon1','salmon2','salmon3','salmon4','salmon5','salmon6','salmon7','salmon8','salmon9','salmon10','salmon11']
    class_names_label = {class_name:i for i, class_name in enumerate(class_names)}
    nb_classes  = len(class_names)
    #print(class_name_label)
    IMAGE_SIZE = (200,100)

    #Modelo que hice inicialmente:
    model_1 = Sequential([
        layers.Rescaling(1./255),
        layers.Conv2D(16,(3, 3), activation = 'relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, (3,3), activation = 'relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation = 'relu'),
        layers.Dense(nb_classes)
    ])

    #Modelo que promone el papers de los salmones pa detectar manchas/puntos:
    model_2 = Sequential([
        layers.Rescaling(1./255),
        layers.Conv2D(8,(3, 3), activation = 'relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(16,(3, 3), activation = 'relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, (6,6), activation = 'relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(1152, activation = 'relu'),
        layers.Dense(nb_classes)
    ])

    #agregar capa de aumentacion de datos, utilisima
    data_augmentation = keras.Sequential(
        [
            layers.RandomBrightness(0.1),
            layers.GaussianNoise(10),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
        ])

    #Modelo con droput y image aumentation
    model_3 = Sequential([
        data_augmentation,
        layers.Rescaling(1./255),
        layers.Conv2D(8, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 6, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Dropout(dropout),
        layers.Flatten(),
        layers.Dense(1152, activation='relu'),
        layers.Dense(nb_classes)
        ])

    model = model_3
    model.compile(optimizer=  optimizer, # adam, sgd, adamax, ...
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    #model.summary()
    return model

def main(optimizer,dropout):
    model = ConvSalmonModel(optimizer,dropout)
    #model.summary()

if __name__ == '__main__':
    main('adam',0.5)
