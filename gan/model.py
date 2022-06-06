import math


from tensorflow import keras
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers as KL
from tensorflow.keras import backend as K
from keras import models as KM
# todo change keras to tf.keras

import ops


def pixel_norm(x):
    return x * tf.math.rsqrt(tf.math.reduce_mean(x, axis=3, keepdims=True) + 1E-8)


def make_kernel(k):
    k = tf.constant(k, dtype=tf.dtypes.float32)

    if k.shape.rank == 1:
        k = k[None, :] * k[:, None]

    k /= tf.math.reduce_sum(k)
    return k


def upsample(x, k, factor=2):
    kernel = make_kernel(k)
    p = kernel.shape[0] - factor

    pad = ((p + 1) // 2 + factor - 1, p // 2)

    return ops.upfirdn2d(x, kernel, up=factor, down=1, pad=pad)


def downsample(x, k, factor=2):
    kernel = make_kernel(k)
    p = kernel.shape[0] - factor

    pad = ((p + 1) // 2, p // 2)

    return ops.upfirdn2d(x, kernel, up=1, down=factor, pad=pad)


def blur(x, k, pad, upsample_factor=1):
    kernel = make_kernel(k)

    if upsample_factor > 1:
        kernel = kernel * (upsample_factor ** 2)

    return ops.upfirdn2d(x, kernel, pad=pad)


class EqualConv2D(KL.Conv2D):
    def __init__(self,
                 filters,
                 kernel_size,
                 strides=(1, 1),
                 padding='valid',
                 data_format=None,
                 dilation_rate=(1, 1),
                 groups=1,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        super().__init__(filters, kernel_size, strides, padding, data_format, dilation_rate, groups, activation,
                         use_bias, kernel_initializer, bias_initializer, kernel_regularizer, bias_regularizer,
                         activity_regularizer, kernel_constraint, bias_constraint, **kwargs)
        self.scale = None

    def build(self, input_shape):
        super(EqualConv2D, self).build(input_shape)
        self.scale = 1 / math.sqrt(self._get_input_channel(input_shape) * self.kernel_size[0] ** 2)

    def call(self, inputs):
        result = self.convolution_op(
            inputs, self.kernel * self.scale
        )
        if self.use_bias:
            result = result + self.bias
        return result


# class EqualLinear(KL.Layer):
#     def __init__(self, units, use_bias=True, bias_init=0, lr_mul=1, activation=None, **kwargs):
#         super(EqualLinear, self).__init__(**kwargs)
#         self.scale = None
#         self.bias = None
#         self.kernel = None
#         self.activation = activation
#         self.units = units
#         self.use_bias = use_bias
#         self.bias_init = bias_init
#         self.lr_mul = lr_mul
#
#     def build(self, input_shape):
#         self.kernel = self.add_weight("kernel", shape=[int(input_shape[-1]), self.units, ],
#                                       initializer=keras.initializers.RandomNormal(mean=0, stddev=1))
#         if self.use_bias:
#             self.bias = self.add_weight("bias", shape=[self.units, ],
#                                         initializer=keras.initializers.Zeros())
#         self.scale = 1 / math.sqrt(input_shape[-1]) * self.lr_mul
#
#     def call(self, inputs, *args, **kwargs):
#         out = tf.matmul(a=inputs, b=self.kernel)
#
#         if self.use_bias:
#             out = tf.nn.bias_add(out, self.bias * self.lr_mul)
#
#         if self.activation:
#             out = keras.activations.relu(out, alpha=0.2) * (2 ** 0.5)
#
#         return out
#
#     def get_config(self):
#         cfg = super(EqualLinear, self).get_config()
#         cfg.update({"units": self.units, "use_bias": self.use_bias, "bias_init": self.bias_init,
#                     "lr_mul": self.lr_mul, "activation": keras.activations.serialize(self.activation)})
#         return cfg


def main():
    inputs = KM.Input(shape=(32, ), name="img", dtype=tf.dtypes.float32)

    outputs = EqualLinear(32)(inputs)

    model = KM.Model(inputs, outputs, name="test")
    model.summary()


if __name__ == "__main__":
    main()
