import tensorflow as tf
import numpy as np

def HomoCoord(coords):
    shape = coords.get_shape().as_list()
    shape[-1] = 1
    ones = tf.ones(shape)
    coords = tf.concat([coords, ones], axis=-1)
    return coords

def ApplyTransform(coords, transform, inverse=False):
    """
    Apply transform to coords
    :param coords:  BxHxWx3
    :param transform:   Bx4x4 or 4x4
    :return:
    """
    batch_size, height, width, _ = coords.get_shape().as_list()

    if len(transform.get_shape()) == 2:
        transform = tf.expand_dims(transform, axis=0)       # 1x4x4
        transform = tf.tile(transform, [batch_size, 1, 1])  # Bx4x4
    if inverse:
        transform = tf.matrix_inverse(transform)

    coords = HomoCoord(coords)  # BxHxWx4
    coords = tf.reshape(coords, [batch_size, height * width, 4])    # BxHWx4
    coords = tf.transpose(coords, [0, 2, 1])    # Bx4xHW
    coords = tf.matmul(transform, coords)   # Bx4xHW
    coords = tf.transpose(coords, [0, 2, 1])    # BxHWx4
    coords = tf.reshape(coords, [batch_size, height, width, 4]) # BxHxWx4
    coords = tf.slice(coords, [0, 0, 0, 0], [-1, -1, -1, 3])    # BxHxWx3
    return coords

def SubSampleImage(image, sample):
    """
    Resize batch of images, get value from the center of grids
    :param image:
    :return:
    """
    batch_size, height, width, channel = image.get_shape().as_list()
    w = np.zeros([sample, sample, channel, channel], dtype=np.float32)
    for i in range(channel):
        w[sample//2, sample//2, i, i] = 1

    resized_image = tf.nn.conv2d(image, w, [1, sample, sample, 1], padding="VALID", name='resize_op')
    return resized_image






