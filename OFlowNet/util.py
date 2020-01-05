import tensorflow as tf
import numpy as np
import math
import tensorflow.contrib.slim as slim

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
    expand = False
    if len(coords.get_shape()) == 3:
        coords = tf.expand_dims(coords, axis=0)
        expand = True
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
    if expand:
        coords = tf.squeeze(coords, axis=0)
    return coords

def GetPixelMap(batch_size, height, width, normalize=False, spec=None):
    """
    Get the normalized pixel coordinates
    :return: BxHxWx2
    """
    map_y = np.zeros((height, width))  # HxW
    map_x = np.zeros((height, width))  # HxW
    for i in range(height):
        map_x[i, :] = range(width)
    for i in range(width):
        map_y[:, i] = range(height)
    map_y = tf.convert_to_tensor(map_y, tf.float32)
    map_y = tf.expand_dims(map_y, axis=-1)
    map_x = tf.convert_to_tensor(map_x, tf.float32)
    map_x = tf.expand_dims(map_x, axis=-1)
    if normalize:
        map_x = (map_x - spec.u) / spec.focal_x
        map_y = (map_y - spec.v) / spec.focal_y
    map = tf.concat([map_x, map_y], axis=-1)  # HxWx2
    map = tf.expand_dims(map, axis=0)  # 1xHxWx2
    map = tf.tile(map, [batch_size, 1, 1, 1])
    return map

########################### data augmentation ##############################
def image_translation_augmentation(image, spec):
    """
    :param image: HxWxC
    :param spec:
    :return H'xW'xC
    :return:
    """
    min_y = 0
    max_y = spec.image_size[0] - spec.crop_size[0]
    min_x = 0
    max_x = spec.image_size[1] - spec.crop_size[1]
    rand_x = 0
    rand_y = 0
    if max_y > min_y:
        rand_y = tf.squeeze(tf.random_uniform([1], min_y, max_y, dtype=tf.int32))
    if max_x > min_x:
        rand_x = tf.squeeze(tf.random_uniform([1], min_x, max_x, dtype=tf.int32))
    with tf.device('/device:CPU:0'):
        image_crop = tf.image.crop_to_bounding_box(image, rand_y, rand_x, spec.crop_size[0], spec.crop_size[1]) # 1xHxWxC
    image_crop = tf.squeeze(image_crop)  # HxWxC
    return image_crop

def image_rotation_augmentation(image, spec):
    batch_size, _, _, _ = image.get_shape().as_list()
    # Rotation
    angle = tf.random_uniform([1], -30, 30, dtype=tf.float32)
    angles = tf.tile(angle, [batch_size])
    radians = angles * math.pi / 180.0
    image = tf.contrib.image.rotate(image, radians, interpolation='NEAREST')
    return image

def image_enlarge_augmentation(image, spec):
    batch_size, _, _, _ = image.get_shape().as_list()

    # Rotation
    image = image_rotation_augmentation(image, spec)

    x1 = tf.squeeze(tf.random_uniform([1], 0, 0.2, dtype=tf.float32))
    y1 = tf.squeeze(tf.random_uniform([1], 0, 0.2, dtype=tf.float32))
    ratio = tf.squeeze(tf.random_uniform([1], 0.8, 1 - tf.maximum(x1, y1), dtype=tf.float32))
    x2 = x1 + ratio
    y2 = y1 + ratio
    boxes = [[y1, x1, y2, x2]] * batch_size
    box_ind = [0] * batch_size
    image = tf.image.crop_and_resize(image, boxes, box_ind, spec.crop_size)  # BxHxWxC
    return image

def image_shrink_augmentation(image, spec):
    batch_size, _, _, _ = image.get_shape().as_list()
    # Rotation
    image = image_rotation_augmentation(image, spec)

    ratio = tf.squeeze(tf.random_uniform([1], 0.8, 1.0))
    new_height = tf.cast(spec.crop_size[0] * ratio, tf.int32)
    new_width = tf.cast(spec.crop_size[1] * ratio, tf.int32)
    resize_image = tf.image.resize_images(image, [new_height, new_width])
    with tf.device('/device:CPU:0'):
        image = tf.image.resize_image_with_crop_or_pad(resize_image, spec.crop_size[0], spec.crop_size[1])
    return image

def image_augmentation(image, spec):
    """
    :param image: HxWxC
    :param spec:
    :return H'xW'xC
    """
    def f1(): return image_translation_augmentation(image, spec)
    def f2(): return image_enlarge_augmentation(image, spec)
    def f3(): return image_shrink_augmentation(image, spec)
    def f4(): return image_rotation_augmentation(image, spec)
    def f5(): return image

    rand_var = tf.squeeze(tf.random_uniform([1], 0, 1., dtype=tf.float32))
    image = tf.case([(rand_var < 0.8, f4)], default=f5, exclusive=False)

    return image

############################ eof data augmentation ##############################

def SSIM(x, y):
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    mu_x = slim.avg_pool2d(x, 3, 1, 'SAME')
    mu_y = slim.avg_pool2d(y, 3, 1, 'SAME')

    sigma_x = slim.avg_pool2d(x ** 2, 3, 1, 'SAME') - mu_x ** 2
    sigma_y = slim.avg_pool2d(y ** 2, 3, 1, 'SAME') - mu_y ** 2
    sigma_xy = slim.avg_pool2d(x * y, 3, 1, 'SAME') - mu_x * mu_y

    SSIM_n = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    SSIM_d = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)

    SSIM = SSIM_n / SSIM_d

    return tf.clip_by_value((1 - SSIM) / 2, 0, 1)

def image_similarity(x, y):
    return 0.85 * SSIM(x, y) + 0.15 * tf.abs(x - y)

def PhotometricLoss(image1, image2, mask=None):
    batch_size, height, width, channel = image1.get_shape().as_list()

    image1 = (image1 / 255.0 * 2.0) - 1
    image2 = (image2 / 255.0 * 2.0) - 1
    loss = image_similarity(image1, image2)

    if mask is not None:
        mask_3 = tf.tile(mask, [1, 1, 1, channel])
        valid_pixel = tf.reduce_sum(mask) + 1.
        loss = mask_3 * loss
    else:
        valid_pixel = batch_size * height * width * channel

    loss = tf.reduce_sum(loss) / valid_pixel
    return loss


