"""Extract visual features from GUI screenshots using a pre-trained ResNet50 model."""

from __future__ import absolute_import, division, print_function
import os
import tensorflow as tf
import numpy as np

from model.classes.model.Config import IMAGE_SIZE

tf.enable_eager_execution()
platform = "web"

IMG_SHAPE = (IMAGE_SIZE, IMAGE_SIZE, 3)

# Create the base model from the pre-trained ResNet50
base_model = tf.keras.applications.ResNet50(input_shape=IMG_SHAPE,
                                               include_top=False,
                                               weights='imagenet')
base_model.trainable = False

for dataset in ("train_data", "val_data", "test_data"):
    for f in os.listdir("../datasets/{}/{}".format(platform, dataset)):
        sample = np.load("../datasets/{}/{}/{}".format(platform, dataset, f))
        input_tensor = sample["img"][np.newaxis, ...]
        output = base_model.apply(input_tensor)
        print(output)
