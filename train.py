import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

import argparse

from model import ConvSalmonModel


def train(opt):
    #inicialiazar el Modelo
    class_names = ['salmon1','salmon2','salmon3','salmon5','salmon7','salmon9','salmon10','salmon11']#['salmon1','salmon2','salmon3','salmon4','salmon5','salmon6','salmon7','salmon8','salmon9','salmon10','salmon11']
    class_names_label = {class_name:i for i, class_name in enumerate(class_names)}
    nb_classes  = len(class_names)
    IMAGE_SIZE = (200,100)
    model = ConvSalmonModel()

    #Entrenar el modelo
    batch_size = opt.batch_size
    epochs = opt.epochs
    data_dir = opt.data_dir #"rois/"

    ## daros de entrenamiento y validacion:
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


    # para guardar checkpoints
    """
    checkpoint_path = str(str(Path(__file__).parent) +"/checkpoints/cp-{epoch:04d}.ckpt")
    checkpoint_dir = os.path.dirname(checkpoint_path)
    # Create a callback that saves the model's weights every 5 epochs
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        verbose=1,
        save_weights_only=True,
        save_freq=100)
    # Save the weights using the `checkpoint_path` format
    """
    if(opt.save):
        checkpoint_filepath = str(str(Path(__file__).parent) +"/checkpoints/best.ckpt")

        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_filepath,
            save_weights_only=True,
            monitor='val_accuracy',
            mode='max',
            save_best_only=True)

        history = model.fit(
          train_ds,
          validation_data=val_ds,
          epochs=epochs,
          callbacks=[model_checkpoint_callback]
        )
    else:
        history = model.fit(
          train_ds,
          validation_data=val_ds,
          epochs=epochs
        )

    # graficos y weas lindas
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

def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default = 32, type=int,help='batch size to train')
    parser.add_argument('--epochs', default = 100 ,type=int,help='epoch to train')
    parser.add_argument('--data_dir', default = "rois/",type=str,help='dir to the dataset')
    parser.add_argument('--save', default = False,type=bool,help='save best checkpoint')

    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt

def main(opt):
	train(opt)

if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
