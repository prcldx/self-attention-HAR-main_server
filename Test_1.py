import os

import h5py
import numpy as np
import tensorflow as tf
import yaml
class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.drop1 = tf.keras.layers.Dropout(rate=0.5)

    def call(self, x, training):
        output = self.drop1(x, training=training)
        return output

    # def forward(self, x, training):
    #     output = self.drop1(x, training=self.training)
    #     return output

x = tf.random.normal((10,10))
#dp1 = tf.keras.layers.Dropout(rate=0.5, training = training)
encode = EncoderLayer()
y = encode(x)
print(y.shape)