from __future__ import absolute_import, division, print_function
import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from model.classes.model.Config import *
tf.enable_eager_execution()
platform = "web"

IMG_SHAPE = (IMAGE_SIZE, IMAGE_SIZE, 3)

# Create the base model from the pre-trained model VGGNET
base_model = tf.keras.applications.ResNet50(input_shape=IMG_SHAPE,
                                               include_top=False,
                                               weights='imagenet')
base_model.trainable = False
print("yes")

for dataset in ("train_data","val_data","test_data"):
    for f in os.listdir("../datasets/{}/{}".format(platform,dataset)):
        sample = np.load("../datasets/{}/{}/{}".format(platform,dataset,f))
        input = sample["img"][np.newaxis,...]
        output = base_model.apply(input)
        print(output)
        break



