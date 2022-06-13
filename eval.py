import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

from model import ConvSalmonModel

class_names = ['salmon1','salmon2','salmon3','salmon5','salmon7','salmon9','salmon10','salmon11']#['salmon1','salmon2','salmon3','salmon4','salmon5','salmon6','salmon7','salmon8','salmon9','salmon10','salmon11']
class_names_label = {class_name:i for i, class_name in enumerate(class_names)}
nb_classes  = len(class_names)
IMAGE_SIZE = (200,100)

#checkpoints


#inicialiazar el Modelo
model = ConvSalmonModel()
model.load_weights(str(str(Path(__file__).parent) +'/checkpoints/cp-0054.ckpt'))

img = tf.keras.utils.load_img(
    str(str(Path(__file__).parent) + '/frames_salmones/salmon1/original/scene00601.png' ) , target_size=(IMAGE_SIZE[0], IMAGE_SIZE[1])
)
img_array = tf.keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0) # Create a batch

predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])

print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)
