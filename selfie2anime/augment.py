import tensorflow as tf

from tensorflow.keras import layers as KL
from tensorflow_addons import image


class RandomAffine(KL.Layer):
    def __init__(self, rotation=0, translation=0, scale=0, shear=0, resize_to=None, crop=None, interpolation="nearest",
                 fill_mode="constant", fill_value=0, **kwargs):
        super(RandomAffine, self).__init__(**kwargs)

        if rotation:
            if isinstance(rotation, (tuple, list)):
                self.rotation = rotation
            else:
                self.rotation = (-rotation, rotation)
        else:
            self.rotation = rotation

        if translation:
            if isinstance(translation, (tuple, list)):
                if len(translation) == 4:
                    self.translation = translation
                else:
                    self.translation = (-translation[0], translation[0], -translation[1], translation[1])
            else:
                self.translation = (-translation, translation, -translation, translation)
        else:
            self.translation = translation

        if scale:
            if isinstance(scale, (tuple, list)):
                if len(scale) == 4:
                    self.scale = scale
                else:
                    self.scale = (scale[0], scale[1], scale[0], scale[1])
            else:
                self.scale = (1 - scale, 1 + scale, 1 - scale, 1 + scale)
        else:
            self.scale = scale

        if shear:
            if isinstance(shear, (tuple, list)):
                if len(shear) == 4:
                    self.shear = shear
                else:
                    self.shear = (-shear[0], shear[0], -shear[1], shear[1])
            else:
                self.shear = (-shear, shear, -shear, shear)
        else:
            self.shear = shear

        self.resize_to = resize_to
        self.crop = crop
        self.interpolation = interpolation
        self.fill_mode = fill_mode
        self.fill_value = fill_value

    def generate_transformations(self, b, h, w, c):
        transformations = []

        if self.rotation:
            angle = tf.random.uniform(shape=(b,), minval=self.rotation[0], maxval=self.rotation[1])
            rotation_transform = image.angles_to_projective_transforms(angle, image_height=h, image_width=w)
            transformations.append(rotation_transform)

        if self.translation:
            dx = tf.random.uniform(shape=(b, 1), minval=self.translation[0], maxval=self.translation[1])
            dy = tf.random.uniform(shape=(b, 1), minval=self.translation[2], maxval=self.translation[3])
            translation_transform = image.translations_to_projective_transforms(tf.concat([dx, dy], axis=1))
            transformations.append(translation_transform)

        if self.scale:
            scale_x = tf.random.uniform(shape=(b, 1), minval=self.scale[0], maxval=self.scale[1])
            scale_y = tf.random.uniform(shape=(b, 1), minval=self.scale[2], maxval=self.scale[3])
            x_offset = ((w - 1.) / 2.0) * (1.0 - scale_x)
            y_offset = ((h - 1.) / 2.0) * (1.0 - scale_y)
            scale_transform = tf.concat([
                scale_x,
                tf.zeros((b, 1), dtype=tf.dtypes.float32),
                x_offset,
                tf.zeros((b, 1), dtype=tf.dtypes.float32),
                scale_y,
                y_offset,
                tf.zeros((b, 1), dtype=tf.dtypes.float32),
                tf.zeros((b, 1), dtype=tf.dtypes.float32)
            ], axis=1)
            transformations.append(scale_transform)

        if self.shear:
            shear_x = tf.random.uniform(shape=(b, 1), minval=self.shear[0], maxval=self.shear[1])
            shear_y = tf.random.uniform(shape=(b, 1), minval=self.shear[2], maxval=self.shear[3])
            shear_transform = tf.concat([
                tf.ones((b, 1), dtype=tf.dtypes.float32),
                shear_x,
                tf.zeros((b, 1), dtype=tf.dtypes.float32),
                shear_y,
                tf.ones((b, 1), dtype=tf.dtypes.float32),
                tf.zeros((b, 1), dtype=tf.dtypes.float32),
                tf.zeros((b, 1), dtype=tf.dtypes.float32),
                tf.zeros((b, 1), dtype=tf.dtypes.float32),
            ], axis=1)
            transformations.append(shear_transform)

        transformation = image.compose_transforms(transformations)
        return transformation

    def call(self, inputs):
        b, h, w, c = inputs.shape
        transformation = self.generate_transformations(b, h, w, c)

        x = image.transform(inputs, transformation, fill_mode=self.fill_mode, interpolation=self.interpolation,
                            fill_value=self.fill_value)

        if self.resize_to:
            x = tf.image.resize(x, size=self.resize_to, method=self.interpolation)
        if self.crop:
            x = tf.image.random_crop(x, (b,) + self.crop + (c,))

        return x


class RandomAffineWithLandmarks(RandomAffine):
    def call(self, inputs):
        img, landmarks = inputs
        b, h, w, c = img.shape

        transformation = self.generate_transformations(b, h, w, c)

        img = image.transform(img, transformation, fill_mode=self.fill_mode, interpolation=self.interpolation,
                              fill_value=self.fill_value)

        if self.resize_to:
            img = tf.image.resize(img, size=self.resize_to, method=self.interpolation)
        if self.crop:
            img = tf.image.random_crop(img, (b,) + self.crop + (c,))

        matrix = tf.reshape(transformation, (-1, 8))
        matrix = tf.reshape(tf.concat((matrix, tf.ones((b, 1))), axis=1), (-1, 1, 3, 3))
        matrix = tf.linalg.inv(matrix)

        landmarks = tf.expand_dims(tf.concat((landmarks, tf.ones((b, landmarks.shape[1], 1))), axis=-1), -1)
        landmarks = tf.squeeze(tf.matmul(matrix, landmarks)[:, :, :2], -1)

        return [img, landmarks]
