import numpy as np
import tensorflow as tf
from keras import layers as KL
from keras import backend as K
from keras import models as KM
from ops import *


def correct_pad(inputs, kernel_size):
    """Returns a tuple for zero-padding for 2D convolution with downsampling.
    Args:
    inputs: Input tensor.
    kernel_size: An integer or tuple/list of 2 integers.
    Returns:
    A tuple.
    """
    img_dim = 2 if K.image_data_format() == 'channels_first' else 1
    input_size = K.int_shape(inputs)[img_dim:(img_dim + 2)]
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    if input_size[0] is None:
        adjust = (1, 1)
    else:
        adjust = (1 - input_size[0] % 2, 1 - input_size[1] % 2)
    correct = (kernel_size[0] // 2, kernel_size[1] // 2)
    return ((correct[0] - adjust[0], correct[0]),
            (correct[1] - adjust[1], correct[1]))


def depth(v, divisor=8, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def se_block(inputs, filters, se_ratio, prefix):
    x = KL.GlobalAvgPool2D(keepdims=True, name=prefix + "squeeze_excite/AvgPool")(inputs)
    x = KL.Conv2D(depth(filters * se_ratio), kernel_size=1, padding='same', name=prefix + "squeeze_excite/Conv")(x)
    x = KL.ReLU(name=prefix + "squeeze_excite/Relu")(x)
    x = KL.Conv2D(filters, kernel_size=1, padding='same', name=prefix + "squeeze_excite/Conv_1")(x)
    x = HardSigmoid(name=prefix + "squeeze_excite/HardSigmoid")(x)
    x = KL.Multiply(name=prefix + "squeeze_excite/Mul")([inputs, x])
    return x


def unet_block(x, shortcut, expansion, filters, kernel_size, stride, se_ratio, activation, block_id, scope=""):
    infilters = x.shape[-1]
    prefix = f"{scope}unet_{block_id}/"

    x = KL.Conv2DTranspose(infilters, kernel_size=3, strides=(2, 2), padding="same",
                           name=prefix + "transpose")(x)
    x = KL.BatchNormalization(momentum=0.99, name=prefix + "transpose/BatchNorm")(x)
    x = activation(name=prefix + "transpose/activation")(x)
    x = tf.concat([x, shortcut], axis=-1)

    return inverted_res_block(x, expansion, filters, kernel_size, 1, se_ratio, activation, block_id, scope)


def inverted_res_block(x, expansion, filters, kernel_size, stride, se_ratio, activation, block_id, scope=""):
    shortcut = x
    prefix = scope + 'expanded_conv/'
    infilters = x.shape[-1]
    if block_id:
        # Expand
        prefix = scope + "expanded_conv_{}/".format(block_id)
        x = KL.Conv2D(depth(infilters * expansion), kernel_size=1, padding="same", use_bias=False,
                      name=prefix + "expand")(x)
        x = KL.BatchNormalization(momentum=0.99, name=prefix + 'expand/BatchNorm')(x)
        x = activation(name=prefix + "expand/activation")(x)

    if stride == 2:
        x = KL.ZeroPadding2D(padding=correct_pad(x, kernel_size), name=prefix + "depthwise/pad")(x)
    x = KL.DepthwiseConv2D(kernel_size, strides=stride, padding="same" if stride == 1 else "valid", use_bias=False,
                           name=prefix + "depthwise")(x)
    x = KL.BatchNormalization(momentum=0.99, name=prefix + 'depthwise/BatchNorm')(x)
    x = activation(name=prefix + "depthwise/activation")(x)

    if se_ratio:
        x = se_block(x, depth(infilters * expansion), se_ratio, prefix)

    x = KL.Conv2D(filters, kernel_size=1, padding="same", use_bias=False, name=prefix + "project")(x)
    x = KL.BatchNormalization(momentum=0.99, name=prefix + "project/BatchNorm")(x)

    if stride == 1 and infilters == filters:
        x = KL.Add(name=prefix + "Add")([shortcut, x])
    return x


def mobilenetv3_large(x, include_preprocessing=False, scope=""):
    activation = HardSwish
    kernel = 5
    se_ratio = 0.25
    if include_preprocessing:
        x = KL.Rescaling(scale=1. / 127.5, offset=-1.)(x)
    # (256, 256, 3) -> (128, 128, 16)
    x = KL.Conv2D(16, kernel_size=3, strides=(2, 2), padding="same", use_bias=False, name=scope + "Conv")(x)
    x = KL.BatchNormalization(momentum=0.99, name=scope + "Conv/BatchNorm")(x)
    x = activation(name=scope + "Conv/activation")(x)

    x = C1 = inverted_res_block(x, 1, depth(16), 3, 1, None, KL.ReLU, 0, scope=scope)
    # (128, 128, 16) -> (64, 64, 24)
    x = inverted_res_block(x, 4, depth(24), 3, 2, None, KL.ReLU, 1, scope=scope)
    x = C2 = inverted_res_block(x, 3, depth(24), 3, 1, None, KL.ReLU, 2, scope=scope)
    # (64, 64, 3) -> (32, 32, 40)
    x = inverted_res_block(x, 3, depth(40), kernel, 2, se_ratio, KL.ReLU, 3, scope=scope)
    x = inverted_res_block(x, 3, depth(40), kernel, 1, se_ratio, KL.ReLU, 4, scope=scope)
    x = C3 = inverted_res_block(x, 3, depth(40), kernel, 1, se_ratio, KL.ReLU, 5, scope=scope)
    # (32, 32, 40) -> (16, 16, 80)
    x = inverted_res_block(x, 6, depth(80), 3, 2, None, activation, 6, scope=scope)
    x = inverted_res_block(x, 2.5, depth(80), 3, 1, None, activation, 7, scope=scope)
    x = inverted_res_block(x, 2.3, depth(80), 3, 1, None, activation, 8, scope=scope)
    x = inverted_res_block(x, 2.3, depth(80), 3, 1, None, activation, 9, scope=scope)
    # (16, 16, 80) -> (16, 16, 112)
    x = inverted_res_block(x, 6, depth(112), 3, 1, se_ratio, activation, 10, scope=scope)
    x = C4 = inverted_res_block(x, 6, depth(112), 3, 1, se_ratio, activation, 11, scope=scope)
    # (16, 16, 112) -> (8, 8, 160)
    x = inverted_res_block(x, 6, depth(160), kernel, 2, se_ratio, activation, 12, scope=scope)
    x = inverted_res_block(x, 6, depth(160), kernel, 1, se_ratio, activation, 13, scope=scope)
    x = inverted_res_block(x, 6, depth(160), kernel, 1, se_ratio, activation, 14, scope=scope)
    return [C1, C2, C3, C4, x]