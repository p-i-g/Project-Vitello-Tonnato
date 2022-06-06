import math

import numpy as np
import tensorflow as tf
from keras import layers as KL
from keras import backend as K
from keras import models as KM
import keras
from keras.engine.input_spec import InputSpec

from ops import *
from mobilenet import depth, se_block


class EqualLinear(KL.Layer):
    def __init__(self, units, use_bias=True, bias_init=0, lr_mul=1, activation=None, **kwargs):
        super(EqualLinear, self).__init__(**kwargs)
        self.scale = None
        self.bias = None
        self.kernel = None
        self.activation = activation
        self.units = units
        self.use_bias = use_bias
        self.bias_init = bias_init
        self.lr_mul = lr_mul

    def build(self, input_shape):
        self.kernel = self.add_weight("kernel", shape=[int(input_shape[-1]), self.units, ],
                                      initializer=keras.initializers.RandomNormal(mean=0, stddev=1))
        if self.use_bias:
            self.bias = self.add_weight("bias", shape=[self.units, ],
                                        initializer=keras.initializers.Constant(self.bias_init))
        self.scale = 1 / math.sqrt(input_shape[-1]) * self.lr_mul

    def call(self, inputs, *args, **kwargs):
        out = tf.matmul(a=inputs, b=self.kernel)

        if self.use_bias:
            out = tf.nn.bias_add(out, self.bias * self.lr_mul)

        if self.activation:
            out = keras.activations.relu(out, alpha=0.2) * (2 ** 0.5)

        return out

    def get_config(self):
        cfg = super(EqualLinear, self).get_config()
        cfg.update({"units": self.units, "use_bias": self.use_bias, "bias_init": self.bias_init,
                    "lr_mul": self.lr_mul, "activation": keras.activations.serialize(self.activation)})
        return cfg


class StyledDepthwiseConv2D(KL.DepthwiseConv2D):
    def __init__(self,
                 kernel_size,
                 strides=(1, 1),
                 padding='valid',
                 depth_multiplier=1,
                 use_style=True,
                 demodulate=True,
                 data_format=None,
                 dilation_rate=(1, 1),
                 activation=None,
                 use_bias=True,
                 depthwise_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 depthwise_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 depthwise_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        super(StyledDepthwiseConv2D, self).__init__(kernel_size=kernel_size,
                                                    strides=strides,
                                                    padding=padding,
                                                    depth_multiplier=depth_multiplier,
                                                    data_format=data_format,
                                                    dilation_rate=dilation_rate,
                                                    activation=activation,
                                                    use_bias=use_bias,
                                                    depthwise_initializer=depthwise_initializer,
                                                    bias_initializer=bias_initializer,
                                                    depthwise_regularizer=depthwise_regularizer,
                                                    bias_regularizer=bias_regularizer,
                                                    activity_regularizer=activity_regularizer,
                                                    depthwise_constraint=depthwise_constraint,
                                                    bias_constraint=bias_constraint, **kwargs)
        self.modulation = None
        self.scale = 0
        self.use_style = use_style
        self.demodulate = demodulate
        self.depthwise_kernel = None
        self.bias = None
        self.input_spec = None

    def build(self, input_shape):
        if len(input_shape[0]) != self.rank + 2:
            raise ValueError('Inputs to `DepthwiseConv` should have '
                             f'rank {self.rank + 2}. '
                             f'Received input_shape={input_shape[0]}.')
        img_shape = tf.TensorShape(input_shape[0])
        style_shape = tf.TensorShape(input_shape[1])
        channel_axis = self._get_channel_axis()
        if img_shape.dims[channel_axis].value is None:
            raise ValueError('The channel dimension of the inputs to `DepthwiseConv` '
                             'should be defined. '
                             f'The input_shape received is {img_shape}, '
                             f'where axis {channel_axis} (0-based) '
                             'is the channel dimension, which found to be `None`.')
        input_dim = int(img_shape[channel_axis])
        depthwise_kernel_shape = self.kernel_size + (input_dim,
                                                     self.depth_multiplier)

        fan_in = input_dim * self.kernel_size[0] ** 2
        self.scale = 1 / math.sqrt(fan_in)

        self.depthwise_kernel = self.add_weight(shape=depthwise_kernel_shape,
                                                initializer=self.depthwise_initializer,
                                                name="depthwise_kernel",
                                                regularizer=self.depthwise_regularizer,
                                                constraint=self.depthwise_constraint)
        if self.use_bias:
            self.bias = self.add_weight(shape=(input_dim * self.depth_multiplier,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)

        if self.use_style:
            self.modulation = EqualLinear(input_dim, bias_init=1)
        else:
            self.modulation = self.add_weight(shape=(1, 1, input_dim, 1),
                                              initializer=keras.initializers.Constant(1))
        self.input_spec = (InputSpec(min_ndim=self.rank + 2, axes={channel_axis: input_dim}),
                           InputSpec(min_ndim=2, axes={1: style_shape[1]}))
        self.built = True

    def call(self, inputs):
        x, style = inputs
        batch, height, width, channels = x.shape

        if self.use_style:
            style = KL.Reshape((1, 1, channels, 1))(self.modulation(style))
            weight = self.scale * self.depthwise_kernel * style
        else:
            weight = self.scale * self.depthwise_kernel * self.modulation

        if self.demodulate:
            demod = tf.math.rsqrt(tf.reduce_sum(tf.pow(weight, 2), [0, 1, 2], keepdims=True))
            weight = weight * demod

        outputs = K.depthwise_conv2d(x,
                                     weight,
                                     strides=self.strides,
                                     padding=self.padding,
                                     dilation_rate=self.dilation_rate,
                                     data_format=self.data_format)
        if self.use_bias:
            outputs = K.bias_add(outputs, self.bias, data_format=self.data_format)
        if self.activation is not None:
            return self.activation(outputs)
        return outputs

    def get_config(self):
        cfg = super(StyledDepthwiseConv2D, self).get_config()
        cfg.update({
            "modulation": self.modulation,
            "scale": self.scale,
            "use_style": self.use_style,
            "demodulate": self.demodulate,
        })
        return cfg


def up_res_block(x, styles, expansion, filters, kernel_size, se_ratio, activation, block_id, upsample=False):
    infilters = x.shape[-1]
    prefix = "up_conv_{}/".format(block_id)

    if upsample:
        x = KL.Conv2DTranspose(infilters, kernel_size=3, strides=(2, 2), padding="same", name=prefix + "transpose")(x)
        x = KL.BatchNormalization(momentum=0.99, name=prefix + "transpose/BatchNorm")(x)
        x = activation(name=prefix + "transpose/activation")(x)
    shortcut = x

    # expand
    x = KL.Conv2D(depth(infilters * expansion), kernel_size=1, padding="same", use_bias=False,
                  name=prefix + "expand")(x)
    x = KL.BatchNormalization(momentum=0.99, name=prefix + 'expand/BatchNorm')(x)
    x = activation(name=prefix + "expand/activation")(x)

    x = StyledDepthwiseConv2D(kernel_size, padding="same", use_bias=False, name=prefix + "depthwise")([x, styles])
    x = KL.BatchNormalization(momentum=0.99, name=prefix + 'depthwise/BatchNorm')(x)
    x = activation(name=prefix + "depthwise/activation")(x)

    if se_ratio:
        x = se_block(x, depth(infilters * expansion), se_ratio, prefix)

    x = KL.Conv2D(filters, kernel_size=1, padding="same", use_bias=False, name=prefix + "project")(x)
    x = KL.BatchNormalization(momentum=0.99, name=prefix + "project/BatchNorm")(x)

    if infilters == filters:
        x = KL.Add(name=prefix + "Add")([shortcut, x])
    return x


def decoder(x, styles):
    activation = HardSwish
    kernel = 5
    se_ratio = 0.25

    x = up_res_block(x, styles, 6, depth(160), kernel, se_ratio, activation, 15)
    x = up_res_block(x, styles, 6, depth(160), kernel, se_ratio, activation, 16)
    x = up_res_block(x, styles, 6, depth(112), kernel, se_ratio, activation, 17, upsample=True)
    x = up_res_block(x, styles, 6, depth(112), kernel, se_ratio, activation, 18)
    x = up_res_block(x, styles, 6, depth(112), kernel, se_ratio, activation, 19)
    x = up_res_block(x, styles, 2.3, depth(80), kernel, se_ratio, activation, 20)
    x = up_res_block(x, styles, 2.3, depth(80), kernel, se_ratio, activation, 21)
    x = up_res_block(x, styles, 2.5, depth(80), kernel, se_ratio, activation, 22)
    x = up_res_block(x, styles, 3, depth(40), kernel, se_ratio, activation, 23, upsample=True)
    x = up_res_block(x, styles, 3, depth(40), kernel, se_ratio, activation, 24)
    x = up_res_block(x, styles, 3, depth(40), kernel, se_ratio, activation, 25)
    x = up_res_block(x, styles, 3, depth(24), kernel, se_ratio, activation, 26, upsample=True)
    x = up_res_block(x, styles, 3, depth(24), kernel, se_ratio, activation, 27)
    x = up_res_block(x, styles, 4, depth(16), kernel, se_ratio, activation, 28)
    x = up_res_block(x, styles, 1, depth(16), kernel, se_ratio, activation, 29)

    x = KL.Conv2D(3, kernel_size=3, padding="same", use_bias=False, name="ToRGB")(x)
    x = KL.BatchNormalization(momentum=0.99, name="ToRGB/BatchNorm")(x)

    return x


if __name__ == "__main__":
    inputs = [KL.Input(shape=(16, 16, 3)), KL.Input(shape=(8,))]
    outputs = StyledDepthwiseConv2D(3, padding="same")(inputs)
    model = KM.Model(inputs, outputs)
    model.summary()
