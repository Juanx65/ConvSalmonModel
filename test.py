import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

import argparse
import seaborn as sns#0.1


from model import ConvSalmonModel

def show_confusion_matrix(cm, labels):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, xticklabels=labels, yticklabels=labels,annot=True, fmt='g')
    plt.xlabel('Prediction')
    plt.ylabel('Label')
    plt.show()

def main(opt):

    label_names = ['salmon1','salmon2','salmon3','salmon4','salmon5','salmon6','salmon7','salmon8','salmon9','salmon10','salmon11']#['salmon1','salmon2','salmon3','salmon5']#['salmon1','salmon2','salmon3','salmon4','salmon5','salmon6','salmon7','salmon8','salmon9','salmon10','salmon11']#
    IMAGE_SIZE = (200,100)

    #Entrenar el modelo
    batch_size = opt.batch_size
    data_dir   = opt.data_dir #"rois/"

    ds_test = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        labels='inferred',
        label_mode='categorical',
        #label_mode='int',
        class_names = label_names,
        seed=None,
        validation_split=None,
        subset=None,
        image_size=(IMAGE_SIZE[0], IMAGE_SIZE[1]),
        batch_size=32)

    model = ConvSalmonModel(opt.optimizer, opt.dropout)
    model.load_weights(str(str(Path(__file__).parent) + opt.weights)).expect_partial()

    labels =  np.array([])
    predictions = np.array([])
    for x, y in ds_test:
      labels = np.concatenate([labels, np.argmax(y.numpy(), axis=-1)])
      predictions = np.concatenate([predictions, np.argmax(model.predict(x), axis = -1)])


    confusion_mtx = tf.math.confusion_matrix(
        labels,
        predictions,
        num_classes=len(label_names))

    show_confusion_matrix(confusion_mtx, label_names)

def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default = 32, type=int,help='batch size to train')
    parser.add_argument('--weights', type=str,default = '/checkpoints/best.ckpt',help='initial weights path')
    parser.add_argument('--data_dir',type=str,default = 'rois_tests/', help='path to dataset to test')
    parser.add_argument('--optimizer', default = 'adam',type=str,help='optimizer for the model ej: adam, sgd, adamax ...')
    parser.add_argument('--dropout', default = 0.8,type=float,help='porcentage para droput de la red, si es que usa')

    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt

if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
