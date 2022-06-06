import numpy as np
import tensorflow as tf
import tensorflow.python.keras.backend as K
import tensorflow.python.keras.layers as KL
import tensorflow.python.keras.models as KM
import tensorflow.python.keras.utils as KU
import tensorflow.python.keras.activations as KA


def autopad(k, p=None):
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(KL.Layer):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True, name="Conv", **kwargs):
        super(Conv, self).__init__(name=name, **kwargs)
        self.pad = KL.ZeroPadding2D(autopad(k, p))
        self.conv = KL.Conv2D(filters=c2, kernel_size=k, strides=s, groups=g, use_bias=False)
        self.bn = KL.BatchNormalization()
        self.act = KL.Activation(KA.swish) if act is True else (act if isinstance(KL.Layer) else KL.Activation(KA.linear))

    def call(self, inputs, *args, **kwargs):
        return self.act(self.bn(self.conv(self.pad(inputs))))

    def forward_fuse(self, inputs):
        return self.act(self.conv(self.pad(inputs)))


class BottleNeck(KL.Layer):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5, name="BottleNeck", **kwargs):
        super(BottleNeck, self).__init__(name=name, **kwargs)
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, k=1, s=1)
        self.cv2 = Conv(c1, c_, k=3, s=1, g=g)
        self.add = shortcut and c1 == c2

    def call(self, inputs, *args, **kwargs):
        x = inputs
        x = self.cv1(x)
        x = self.cv2(x)
        if self.add:
            x = x + inputs
        return x


class C3(KL.Layer):
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, name="C3", **kwargs):
        super(C3, self).__init__(name=name, **kwargs)
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, k=1, s=1)
        self.cv2 = Conv(c1, c_, k=1, s=1)
        self.cv3 = Conv(2 * c_, c2, k=1)
        self.m = KM.Sequential(*[BottleNeck(c_, c_, shortcut=shortcut, g=g, e=e) for _ in range(n)])

    def call(self, inputs, *args, **kwargs):
        x = inputs
        x1 = self.cv2(x)
        x = self.cv1(x)
        x = self.m(x)
        x = tf.concat((x, x1), axis=4)  # channels last
        x = self.cv3(x)
        return x


class SPP(KL.Layer):
    # Spatial Pyramid Pooling (SPP) layer https://arxiv.org/abs/1406.4729
    def __init__(self, c1, c2, k=(5, 9, 13), name="SPP", **kwargs):
        super(SPP, self).__init__(name=name, **kwargs)
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, k=1, s=1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, k=1, s=1)
        self.m = [KM.Sequential([KL.ZeroPadding2D(x), KL.MaxPooling2D(pool_size=x, strides=1)]) for x in k]

    def call(self, inputs, *args, **kwargs):
        x = self.cv1(inputs)
        x = tf.concat([x] + [m(x) for m in self.m], axis=4)  # channels last
        return self.cv2(x)


class Focus(KL.Layer):
    # Focus wh information into c-space
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True, name="Focus", **kwargs):
        super(Focus, self).__init__(name=name, **kwargs)
        self.conv = Conv(c1 * 4, c2, k=k, s=s, p=p, g=g, act=act)

    def call(self, inputs, *args, **kwargs):
        x = tf.concat([input[..., ::2, ::2], input[..., 1::2, ::2], input[..., ::2, 1::2], input[..., 1::2, 1::2]])
        return self.conv(x)
