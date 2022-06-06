from common import *


class Detect(KL.Layer):
    stride = None  # stride computed during build
    onnx_dynamic = False  # ONNX export parameter

    def __init__(self, nc=80, anchors=(), ch=(), inplace=True, num_coords=0, name="Detect", **kwargs):
        super(Detect, self).__init__(name=name, **kwargs)
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchors
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [tf.zeros(1)] * self.nl  # init grid
        self.anchors = tf.reshape(tf.constant(anchors), (self.nl, -1, 2))
        self.anchor_grid = tf.reshape(tf.identity(self.anchors), (self.nl, 1, -1, 1, 1, 2))
        self.m = [KL.Conv2D(self.no * self.na, 1) for x in ch]  # output conv
        self.inplace = inplace  # use inplace operations
        self.num_coords = num_coords

    def call(self, inputs, *args, training=True, **kwargs):
        z = []  # inference output
        for i in range(self.nl):
            inputs[i] = self.m[i](inputs[i])  # conv
            bs, ny, nx, _ = inputs[i].shape  # x(bs, 20, 20, 255) to x(bs, 3, 20, 20, 85)
            inputs[i] = tf.transpose(tf.reshape(inputs[i], (bs, ny, nx, self.na, self.no)), (0, 3, 1, 2, 4))

            if not training:
                if self.grid[i].shape[2:4] != inputs[i].shape[2:4] or self.onnx_dynamic:
                    self.grid[i] = self._make_grid(nx, ny)

                    y = KA.sigmoid(inputs)
                    if self.inplace:  # i don't know what this is doing
                        y[..., 0:2] = (y[..., 0:2] * 2 - 0.5 + self.grid[i]) * self.stride[i]  # xy
                        y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh

                        if hasattr(self, "num_coords") and self.num_coords:
                            y[..., -self.num_coords:] = y[..., -self.num_coords:] * 4. - 2.
                            y[..., -self.num_coords:] *= tf.tile(self.anchor_grid[i], (1, 1, 1, 1, self.num_coords // 2))
                            y[..., -self.num_coords:] += tf.tile((self.grid[i] * self.stride[i]), (1, 1, 1, 1, self.num_coords // 2))

                    else:
                        xy = (y[..., 0:2] * 2 - 0.5 + self.grid[i]) * self.stride[i]  # xy
                        wh = (y[..., 2:4] * 2) ** 2 * tf.reshape(self.anchor_grid[i], (1, self.na, 1, 1, 2))  # wh
                        y = tf.concat((xy, wh, y[..., 4:]), axis=-1)
                    z.append(tf.reshape(y, (bs, -1, self.no)))
            return inputs if self.training else (tf.concat(z, axis=1), inputs)

    @staticmethod
    @tf.function
    def _make_grid(nx=20, ny=20):
        xv, yv = tf.meshgrid([tf.range(nx), tf.range(ny)])
        return tf.cast(tf.reshape(tf.stack((xv, yv), 2), (1, 1, ny, nx)), tf.float32)
