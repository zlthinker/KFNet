import tensorflow as tf
import numpy as np
import math
import matplotlib.pyplot as plt
import scipy.stats

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

def FlowFromCoordMap(coord_map1, pose2, spec):
    """
    :param coord_map1: BxHxWx4 - coordinate map of camera1
    :param pose2: Bx3x3 - camera pose of camera2, world space to camera space
    :return: BxHxWx3 - optical flow with mask from camera1 to camera2
    """

    batch_size, height, width, _ = coord_map1.get_shape().as_list()
    mask1 = tf.slice(coord_map1, [0, 0, 0, 3], [-1, -1, -1, 1])         # BxHxWx1
    coord_map1 = tf.slice(coord_map1, [0, 0, 0, 0], [-1, -1, -1, 3])    # BxHxWx3
    coord_map1 = tf.reshape(coord_map1, [batch_size, -1, 3])            # BxHWx3
    coord_map1 = tf.transpose(coord_map1, [0, 2, 1])                    # Bx3xHW

    camera_coord_map1 = tf.matmul(pose2, coord_map1)                    # Bx3xHW
    camera_coord_map1 = tf.transpose(camera_coord_map1, [0, 2, 1])      # BxHWx3
    camera_coord_map1 = tf.reshape(camera_coord_map1, [batch_size, height, width, 3])   # BxHxWx3
    camera_coord_map1_x = tf.slice(camera_coord_map1, [0, 0, 0, 0], [-1, -1, -1, 1])    # BxHxWx1
    camera_coord_map1_y = tf.slice(camera_coord_map1, [0, 0, 0, 1], [-1, -1, -1, 1])
    camera_coord_map1_z = tf.slice(camera_coord_map1, [0, 0, 0, 2], [-1, -1, -1, 1])
    camera_pixel_map2_x = tf.divide(camera_coord_map1_x, camera_coord_map1_z) * spec.focal_x + spec.u   # BxHxWx1
    camera_pixel_map2_y = tf.divide(camera_coord_map1_y, camera_coord_map1_z) * spec.focal_y + spec.v
    camera_pixel_map2 = tf.concat([camera_pixel_map2_x, camera_pixel_map2_y], axis=-1)  # BxHxWx2

    camera_pixel_map1 = GetPixelMap(batch_size, height, width, False)   # BxHxWx2
    optical_flow = camera_pixel_map2 - camera_pixel_map1            # BxHxWx2
    optical_flow = tf.concat([optical_flow, mask1], axis=-1)        # BxHxWx3
    return optical_flow

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

def image_enlarge_augmentation(image, spec):
    batch_size, _, _, _ = image.get_shape().as_list()

    # Rotation
    angle = tf.random_uniform([1], -30, 30, dtype=tf.float32)
    angles = tf.tile(angle, [batch_size])
    radians = angles * math.pi / 180.0
    image = tf.contrib.image.rotate(image, radians, interpolation='NEAREST')

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
    angle = tf.random_uniform([1], -30, 30, dtype=tf.float32)
    angles = tf.tile(angle, [batch_size])
    radians = angles * math.pi / 180.0
    image = tf.contrib.image.rotate(image, radians, interpolation='NEAREST')

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

    rand_var = tf.squeeze(tf.random_uniform([1], 0, 1., dtype=tf.float32))
    image = tf.case([(rand_var < 0.1, f1), (rand_var < 0.55, f2)], default=f3, exclusive=False)

    return image

############################ eof data augmentation ##############################

############################ visualization ##############################
def plot_pdf(data_array, save_path=None):
    # clear zeros
    data_array = filter(lambda x: x > 0 and x < 20, data_array)

    min_val = min(data_array)
    max_val = max(data_array)
    bin_size = 0.01
    bin_num = int((max_val - min_val) / bin_size)

    fig, ax1 = plt.subplots()
    vals, bin_edges, patches = ax1.hist(data_array, bin_num, normed=1, facecolor='b', alpha=0.5)
    bw = 0.1
    t_range = np.linspace(min_val, max_val, 1000)
    kde = scipy.stats.gaussian_kde(data_array, bw_method=bw)
    ax1.plot(t_range, kde(t_range), lw=2, color='b')
    ax1.set_ylabel('PDF')

    vals /= (bin_size * vals).sum()
    C_vals = np.cumsum(np.asarray(vals * bin_size))
    c_range = bin_edges[0:-1]
    ax2 = ax1.twinx()
    ax2.plot(c_range, C_vals, color='r')
    ax2.set_ylabel('CDF')

    # area under curve
    AUC = np.sum(C_vals) * bin_size / (max_val - min_val)
    plt.title('AUC='+str(AUC))
############################ eof visualization ##############################

def RandomRotation():
    """
    Generate a random 3x3 rotatio matrix
    :return: 3x3
    :return: 3x3
    """
    angle_x = tf.squeeze(tf.random_uniform([1], 0, 2 * math.pi, dtype=tf.float32))
    angle_y = tf.squeeze(tf.random_uniform([1], 0, 2 * math.pi, dtype=tf.float32))
    angle_z = tf.squeeze(tf.random_uniform([1], 0, 2 * math.pi, dtype=tf.float32))

    rot_x = tf.stack([1, 0, 0, 0, tf.cos(angle_x), -tf.sin(angle_x), 0, tf.sin(angle_x), tf.cos(angle_x)])
    rot_y = tf.stack([tf.cos(angle_y), 0, tf.sin(angle_y), 0, 1, 0, -tf.sin(angle_y), 0, tf.cos(angle_y)])
    rot_z = tf.stack([tf.cos(angle_z), -tf.sin(angle_z), 0, tf.sin(angle_z), tf.cos(angle_z), 0, 0, 0, 1])
    rot_x = tf.reshape(rot_x, [3, 3])
    rot_y = tf.reshape(rot_y, [3, 3])
    rot_z = tf.reshape(rot_z, [3, 3])

    rot = tf.matmul(rot_z, rot_y)
    rot = tf.matmul(rot, rot_x)
    return rot

def RotateCoord(coord_map, rotation):
    """
    :param coord_map: BxHxWx3
    :param rotation: 3x3
    :return: BxHxWx3
    """
    batch_size, height, width, _ = coord_map.get_shape().as_list()

    coord_map = tf.reshape(coord_map, [batch_size, -1, 3])  # BxHWx3

    rotation = tf.expand_dims(rotation, axis=0) # 1x3x3
    rotation = tf.tile(rotation, [batch_size, 1, 1])    # Bx3x3
    coord_map = tf.matmul(coord_map, rotation) # BxHWx3
    coord_map = tf.reshape(coord_map, [batch_size, height, width, 3])
    return coord_map

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
        # w[3, 4, i, i] = 0.25
        # w[4, 3, i, i] = 0.25
        # w[4, 4, i, i] = 0.25
    # kernel = tf.Variable(w, trainable=False)

    resized_image = tf.nn.conv2d(image, w, [1, sample, sample, 1], padding="VALID", name='resize_op')
    return resized_image