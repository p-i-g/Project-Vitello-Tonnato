from tensorflow.keras import utils as KU
from tensorflow.keras import models as KM
from ops import *
import mobilenet


def discriminator(x):
    kernel_size = 5
    se_ratio = 0.25
    activation = HardSwish

    C1, C2, C3, C4, C5 = mobilenet.mobilenetv3_large(x)
    l_adv = mobilenet.inverted_res_block(C3, 2.3, mobilenet.depth(80), kernel_size, 1, se_ratio, activation, 15,
                                         scope="discriminator/l_branch/")

    g_branch = C5
    g_adv = KL.Conv2D(1, 1)(g_branch)

    g_std = KL.Flatten()(g_branch)
    g_std = KL.Dense(128)(g_std)
    g_std = HardSwish()(g_std)
    stddev = tf.math.reduce_std(g_std)
    g_std = tf.broadcast_to(stddev, tf.shape(g_std))
    g_std = KL.Dense(1)(g_std)

    return l_adv, g_adv, g_std


if __name__ == "__main__":
    inputs = KL.Input(shape=(256, 256, 3))
    outputs = discriminator(inputs)
    model = KM.Model(inputs, outputs)
    KU.plot_model(model)
