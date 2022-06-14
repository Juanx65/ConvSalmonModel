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

def eval(opt):
    class_names = ['salmon1','salmon2','salmon3','salmon4','salmon5','salmon6','salmon7','salmon8','salmon9','salmon10','salmon11']#['salmon1','salmon2','salmon3','salmon5','salmon7','salmon9','salmon10','salmon11']#['salmon1','salmon2','salmon3','salmon4','salmon5','salmon6','salmon7','salmon8','salmon9','salmon10','salmon11']
    class_names_label = {class_name:i for i, class_name in enumerate(class_names)}
    nb_classes  = len(class_names)
    IMAGE_SIZE = (200,100)

    #inicialiazar el Modelo
    model = ConvSalmonModel(opt.optimizer, opt.dropout)
    model.load_weights(str(str(Path(__file__).parent) + opt.weights)).expect_partial()#'/checkpoints/cp-0100.ckpt'))

    img = tf.keras.utils.load_img(
        str(str(Path(__file__).parent) + opt.image) , target_size=(IMAGE_SIZE[0], IMAGE_SIZE[1])#'/frames_salmones/salmon5/original/scene00161.png' ) , target_size=(IMAGE_SIZE[0], IMAGE_SIZE[1])
    )
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    print(
        "This image most likely belongs to {} with a {:.2f} percent confidence."
        .format(class_names[np.argmax(score)], 100 * np.max(score))
    )

def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str,help='initial weights path')
    parser.add_argument('--image',type=str,help='path image to evaluate')
    parser.add_argument('--optimizer', default = 'adam',type=str,help='optimizer for the model ej: adam, sgd, adamax ...')
    parser.add_argument('--dropout', default = 0.8,type=float,help='porcentage para droput de la red, si es que usa')

    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt

def main(opt):
	eval(opt)

if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
