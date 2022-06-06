import tensorflow as tf
from keras import layers as KL


class HardSigmoid(KL.Layer):
    def call(self, inputs, *args, **kwargs):
        return tf.nn.relu6(inputs + 3.) * (1. / 6.)


class HardSwish(KL.Layer):
    def call(self, inputs, *args, **kwargs):
        return inputs * tf.nn.relu6(inputs + 3.) * (1. / 6.)