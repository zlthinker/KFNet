import math
from cnn_wrapper import helper, ScoreNet
from loss import *
from util import *

FLAGS = tf.app.flags.FLAGS

# Params for config.
tf.app.flags.DEFINE_string('save_dir', './ckpt', """Path to save the model.""")
tf.app.flags.DEFINE_boolean('is_training', True, """Flag to training model""")
tf.app.flags.DEFINE_boolean('debug', False, """Flag to debug""")
tf.app.flags.DEFINE_boolean('show', False, """Flag to turn on visualization""")
tf.app.flags.DEFINE_boolean('shuffle', False, """Flag to shuffle data""")
tf.app.flags.DEFINE_integer('gpu', 0, """GPU id.""")

# Params for solver.
tf.app.flags.DEFINE_float('base_lr', 0.0001, """Base learning rate.""")
tf.app.flags.DEFINE_integer('max_steps', 800000, """Max training iteration.""")
tf.app.flags.DEFINE_integer('max_epoch', 600, """Max epoch.""")
tf.app.flags.DEFINE_integer('display', 10, """Interval of loginfo display.""")
tf.app.flags.DEFINE_integer('stepvalue', 50000, """Step interval to decay learning rate.""")
tf.app.flags.DEFINE_integer('snapshot', 5000, """Step interval to save the model.""")
tf.app.flags.DEFINE_float('gamma', 0.5, """Learning rate decay rate.""")
tf.app.flags.DEFINE_float('weight_decay', 0.0001, """Fraction of regularization term.""")
tf.app.flags.DEFINE_float('momentum', 0.9, """momentum in when using SGD with momentum solver""")
tf.app.flags.DEFINE_integer('reset_step', -1, """Reset step.""")

# Params for I/O
tf.app.flags.DEFINE_string('input_folder', '', """Path to data.""")
tf.app.flags.DEFINE_string('model_folder', '', """Path to model.""")
tf.app.flags.DEFINE_string('finetune_folder', '', """Path to model from which we fine-tune it.""")
tf.app.flags.DEFINE_string('output_folder', '', """Path to output.""")
tf.app.flags.DEFINE_string('scene', '', """Path to save the model.""")


def get_transform():
    if FLAGS.scene == 'chess':
        transform = tf.constant([[0.95969, 0.205793, -0.191428, 0.207924],
                                 [0.0738071, 0.47266, 0.878149, -0.190767],
                                 [0.271197, -0.856879, 0.438418, 1.71634],
                                 [0, 0, 0, 1]])
    elif FLAGS.scene == 'fire':
        transform = tf.constant([[0.999908, 0.0125542, -0.00509274, -0.306473],
                                 [0.0110424, -0.537429, 0.843236, 0.297648],
                                 [-0.00784919, 0.843215, 0.537519, 1.73969],
                                 [0, 0, 0, 1]])
    elif FLAGS.scene == 'heads':
        transform = tf.constant([[0.982153, 0.159748, 0.0992782, 0.20507],
                                 [0.0660033, -0.787006, 0.613405, 0.0496231],
                                 [-0.176123, 0.595904, 0.783504, 0.829125],
                                 [0, 0, 0, 1]])
    elif FLAGS.scene == 'office':
        transform = tf.constant([[0.995655, 0.0930714, -0.00300323, -0.188047],
                                 [-0.039624, 0.452633, 0.890816, -0.473323],
                                 [0.0842688, -0.886826, 0.454354, 2.31595],
                                 [0, 0, 0, 1]])
    elif FLAGS.scene == 'pumpkin':
        transform = tf.constant([[0.984219, 0.0694752, -0.162745, -0.489651],
                                 [-0.0569574, 0.995137, 0.080364, -1.41071],
                                 [0.167537, -0.0698263, 0.98339, 2.32241],
                                 [0, 0, 0, 1]])
    elif FLAGS.scene == 'redkitchen':
        transform = tf.constant([[0.968627, 0.248297, -0.0104599, -0.610452],
                                 [-0.130742, 0.544927, 0.828228, -0.337849],
                                 [0.211346, -0.800876, 0.560294, 2.50185],
                                 [0, 0, 0, 1]])
    elif FLAGS.scene == 'stairs':
        transform = tf.constant([[0.157028, -0.980771, 0.115889, 0.0705374],
                                 [-0.562567, 0.00761531, 0.826717, -0.55947],
                                 [0.811702, 0.195013, 0.550554, 2.30206],
                                 [0, 0, 0, 1]])
    elif FLAGS.scene == 'apt1-kitchen':
        transform = tf.constant([[-0.439549,	0.877178,	0.193273,	-2.53131],
                                 [0.853136,	0.475022,	-0.21567,	-2.36806],
                                 [0.28099,	-0.0700909,	0.957148,	0.961991],
                                 [0,	0,	0,	1]])
    elif FLAGS.scene == 'apt1-living':
        transform = tf.constant([[-0.976857,	0.196845,	0.0836852,	-0.253685],
                                [0.198407,	0.980061,	0.0106963,	1.80821],
                                [0.0799111,	-0.0270524,	0.996435,	0.486379],
                                [0,	0,	0,	1]])
    elif FLAGS.scene == 'apt2-bed':
        transform = tf.constant([[0.498837,	-0.863166,	0.0781468,	0.657568],
                                [0.86455,	0.501916,	0.0251732,	-3.99714],
                                [-0.0609518,	0.0550045,	0.996624,	0.321481],
                                [-0,	-0,	0,	1	]])
    elif FLAGS.scene == 'apt2-kitchen':
        transform = tf.constant([[0.309051,	0.951024,	-0.00639379,	-0.216256	],
                                [-0.946868,	0.308316,	0.0915578,	-0.0881695	],
                                [0.089045,	-0.022242,	0.995779,	1.00013	],
                                [-0,	-0,	0,	1]])
    elif FLAGS.scene == 'apt2-living':
        transform = tf.constant([[0.728244,	0.685249,	-0.00971653,	0.103978	],
                                [0.684718,	-0.726941,	0.0521262,	-0.367157	],
                                [-0.028656,	0.0446137,	0.998593,	0.415195	],
                                [0,	0,	-0,	1	]])
    elif FLAGS.scene == 'apt2-luke':
        transform = tf.constant([[0.194524,	0.978697,	0.0656685,	0.481434	],
                                [0.97906,	-0.19782,	0.0480407,	3.76976	],
                                [0.0600078,	0.0549483,	-0.996684,	0.777835	],
                                [-0,	-0,	0,	1	]])
    elif FLAGS.scene == 'office1-gates362':
        transform = tf.constant([[0.975627,	0.185328,	0.117494,	-5.83698],
                                [0.182211,	-0.982569,	0.0368314,	-0.885536],
                                [-0.122272,	0.014525,	0.99239,	0.708184],
                                [0,	0,	-0,	1]])
    elif FLAGS.scene == 'office1-gates381':
        transform = tf.constant([[-0.0834194,	0.99637,	0.0169646,	5.10064	],
                                [0.996148,	0.0829149,	0.028538,	2.18725	],
                                [0.0270278,	0.0192799,	-0.999449,	0.567404],
                                [0,	0,	-0,	1]])
    elif FLAGS.scene == 'office1-lounge':
        transform = tf.constant([[-0.963156,	0.26642,	0.0367585,	-0.123225	],
                                [0.267187,	0.963482,	0.0177123,	0.76595],
                                [0.0306973,	-0.0268811,	0.999167,	0.521955],
                                [0,	0,	-0,	1]])
    elif FLAGS.scene == 'office1-manolis':
        transform = tf.constant([[0.485911,	-0.869785,	0.0858218,	0.302229],
                                [0.869861,	0.471713,	-0.144319,	-0.0445168],
                                [0.0850431,	0.144779,	0.985803,	0.811429],
                                [0,	0,	-0,	1]])
    elif FLAGS.scene == 'office2-5a':
        transform = tf.constant([[0.963581,	-0.267127,	0.0124365,	0.0302841	],
                                [0.267036,	0.958687,	-0.0980343,	6.88496	],
                                [0.0142649,	0.097785,	0.995105,	0.389961	],
                                [-0,	-0,	0,	1	]])
    elif FLAGS.scene == 'office2-5b':
        transform = tf.constant([[-0.127953,	0.978769,	0.16012,	0.899368	],
                                [0.990278,	0.117201,	0.0749226,	0.658585	],
                                [0.0545658,	0.16815,	-0.98425,	0.602684	],
                                [-0,	-0,	0,	1	]])
    elif FLAGS.scene == 'GreatCourt':
        transform = tf.constant([[0.935751,	0.352355,	0.0147003,	55.2724	],
                                 [-0.352448,	0.935822,	0.00417758,	35.9317	],
                                 [0.0122848,	0.00909025,	-0.999883,	3.44179	],
                                 [0, 0, 0, 1]])
    elif FLAGS.scene == 'KingsCollege':
        transform = tf.constant([[0.995572,	0.0859924,	0.0379575,	13.1962	],
                                 [-0.0939971,	0.911174,	0.401155,	-3.90996],
                                 [8.9566e-05,	0.402947,	-0.915223,	4.02919	],
                                 [0, 0, 0, 1]])
    elif FLAGS.scene == 'OldHospital':
        transform = tf.constant([[0.999639,	0.017642,	-0.0202718,	14.8957	],
                                 [-0.00206426,	0.802522,	0.596619,	8.62598	],
                                 [0.0267941,	-0.596361,	0.802269,	1.61564],
                                 [0, 0, 0, 1]])
    elif FLAGS.scene == 'ShopFacade':
        transform = tf.constant([[-0.345171,	0.879857,	0.326664,	1.5323	],
                                 [0.937818,	0.309693,	0.156805,	3.90066	],
                                 [0.0368007,	0.360476,	-0.932042,	2.10786	],
                                 [0, 0, 0, 1]])
    elif FLAGS.scene == 'StMarysChurch':
        transform = tf.constant([[0.142238,	-0.989832,	0.000267895,	12.2121	],
                                 [0.0355632,	0.00538086,	0.999353,	-4.4018	],
                                 [0.989193,	0.142136,	-0.035967,	20.3682],
                                 [0, 0, 0, 1]])
    elif FLAGS.scene == 'Street':
        transform = tf.constant([[-0.996538,	0.0830094,	0.00463003,	-40.8375],
                                 [0.0831384,	0.994957,	0.0561204,	4.67856],
                                 [5.18426e-05,	0.0563111,	-0.998413,	2.3792],
                                 [0, 0, 0, 1]])
    elif FLAGS.scene == 'Street-east':
        transform = tf.constant([[0.998489,	0.0543086,	0.00838941,	-3.0179],
                                 [-0.0549527,	0.986745,	0.152692,	-5.84657],
                                 [1.4293e-05,	-0.152923,	0.988238,	2.37806],
                                 [0, 0, 0, 1]])
    elif FLAGS.scene == 'Street-west':
        transform = tf.constant([[-0.996692,	0.0809217,	0.00759483,	-10.445],
                                 [0.0812663,	0.990659,	0.109499,	11.4673],
                                 [0.00133692,	0.109754,	-0.993958,	2.81858],
                                 [0, 0, 0, 1]])
    elif FLAGS.scene == 'Street-south':
        transform = tf.constant([[0.997679,	0.0678975,	0.00509212,	-21.7206],
                                 [-0.0680618,	0.996581,	0.0468482,	2.18078],
                                 [0.00189384,	0.047086,	-0.998889,	2.10013],
                                 [0, 0, 0, 1]])
    elif FLAGS.scene == 'Street-north1':
        transform = tf.constant([[0.998512,	0.0544589,	0.00289331,	-6.68181],
                                 [-0.0545117,	0.998241,	0.0233179,	-0.955109],
                                 [0.00161835,	0.0234409,	-0.999724,	2.93765	],
                                 [0, 0, 0, 1]])
    elif FLAGS.scene == 'Street-north2':
        transform = tf.constant([[0.996588,	0.0820581,	0.00882958,	2.80514	],
                                 [-0.0822394,	0.996356,	0.0226144,	-0.00355728],
                                 [0.00694171,	0.0232633,	-0.999705,	2.94198],
                                 [0, 0, 0, 1]])
    elif FLAGS.scene == 'DeepLoc':
        transform = tf.constant([[-0.60111,	0.799091,	0.0109981,	-0.0550447	],
                                 [0.00873532,	-0.00719131,	0.999936,	-0.039275],
                                 [0.799119,	0.601168,	-0.00265754,	-0.138549],
                                 [0, 0, 0, 1]])
    else:
        print 'Invalid scene:', FLAGS.scene
        exit()
    return transform

def get_indexes(num, is_training):
    indexes = []
    blind_indexes = []
    if FLAGS.scene == 'chess':
        indexes = range(num)
        blind_indexes = range(2924, 2944)
    elif FLAGS.scene == 'fire':
        indexes = range(num)
    elif FLAGS.scene == 'heads':
        indexes = range(num)
    elif FLAGS.scene == 'office':
        indexes = range(num)
    elif FLAGS.scene == 'pumpkin':
        indexes = range(num)
    elif FLAGS.scene == 'redkitchen':
        indexes = range(num)
    elif FLAGS.scene == 'stairs':
        indexes = range(num)
    elif FLAGS.scene == 'apt1-kitchen':
        if is_training:
            indexes = range(357, num)
        else:
            indexes = range(357)
    elif FLAGS.scene == 'apt1-living':
        if is_training:
            indexes = range(493, num)
        else:
            indexes = range(493)
    elif FLAGS.scene == 'apt2-bed':
        if is_training:
            indexes = range(244, num)
        else:
            indexes = range(50) + range(90, 244)
        blind_indexes += range(50, 90)
        blind_indexes += range(244, 250)
        blind_indexes += range(700, 710)
        blind_indexes += range(735, 741)
        blind_indexes += range(904, 910)
    elif FLAGS.scene == 'apt2-kitchen':
        if is_training:
            indexes = range(230, num)
        else:
            indexes = range(230)
        blind_indexes += range(10)
        blind_indexes += range(50, 60)
        blind_indexes += range(386, 390)
        blind_indexes += range(590, 600)
    elif FLAGS.scene == 'apt2-living':
        if is_training:
            indexes = range(359, num)
        else:
            indexes = range(359)
        blind_indexes += range(10)
        blind_indexes += [359]
        blind_indexes += range(745, 750)
    elif FLAGS.scene == 'apt2-luke':
        if is_training:
            indexes = range(624, num)
        else:
            indexes = range(624)
    elif FLAGS.scene == 'office1-gates362':
        if is_training:
            indexes = range(386, num)
        else:
            indexes = range(386)
        blind_indexes += range(1327, 1330)
        blind_indexes += [2279]
    elif FLAGS.scene == 'office1-gates381':
        if is_training:
            indexes = range(1053, num)
        else:
            indexes = range(1053)
    elif FLAGS.scene == 'office1-lounge':
        if is_training:
            indexes = range(330, 485)
            indexes += range(490, num)
        else:
            indexes = range(327)
    elif FLAGS.scene == 'office1-manolis':
        if is_training:
            indexes = range(807, num)
        else:
            indexes = range(807)
        blind_indexes += range(10)
        blind_indexes += range(1533, 1540)
    elif FLAGS.scene == 'office2-5a':
        if is_training:
            indexes = range(497, num)
        else:
            indexes = range(497)
    elif FLAGS.scene == 'office2-5b':
        if is_training:
            indexes = range(415, num)
        else:
            indexes = range(415)
        blind_indexes += range(10)
        blind_indexes += range(415, 441)
        blind_indexes += range(933, 951)
        blind_indexes += range(1392, 1400)
    elif FLAGS.scene == 'GreatCourt':
        indexes = range(num)
    elif FLAGS.scene == 'KingsCollege':
        indexes = range(num)
    elif FLAGS.scene == 'OldHospital':
        indexes = range(num)
    elif FLAGS.scene == 'ShopFacade':
        indexes = range(num)
    elif FLAGS.scene == 'StMarysChurch':
        indexes = range(num)
    elif FLAGS.scene == 'Street':
        indexes = range(num)
    elif FLAGS.scene == 'Street-east' or FLAGS.scene == 'Street-west' or FLAGS.scene == 'Street-south' \
            or FLAGS.scene == 'Street-north1' or FLAGS.scene == 'Street-north2':
        indexes = range(num)
    elif FLAGS.scene == 'DeepLoc':
        indexes = range(num)
    else:
        print 'Invalid scene:', FLAGS.scene
        exit()
    indexes = [i for i in indexes if i not in blind_indexes]
    return indexes


def softmax_pooling(input_tensor):
    """
    :param input_tensor: shape [NxHxWxC]
    :return: shape [NxHxW]
    """
    pool_conf = tf.layers.max_pooling2d(input_tensor, 5, 1, padding='same', name='pool_conf')
    tanh_conf = tf.tanh(pool_conf)
    exp = tf.exp(tanh_conf)
    pool_size = 32
    avg = tf.layers.average_pooling2d(exp, pool_size, 1, padding='same', name='avg_conf')
    sum = avg * pool_size * pool_size
    conf = tf.squeeze(tf.div(exp, sum))
    return conf

def get_pixel_map(height, width, spec=None):
    """
    Get the normalized pixel coordinates
    :return: HxWx2
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

    if spec is not None:
        map_y = (map_y - spec.v) / spec.focal_y
        map_x = (map_x - spec.u) / spec.focal_x

    map = tf.concat([map_x, map_y], axis=-1) # HxWx2
    return map

def get_weighted_square_coefficient(coefficient_map, weight_map, spec, name=None):
    """
    :param coefficient_map: BxHxWx12
    :param weight_map: BxHxWx1
    :return: Bx12x12
    """
    tile_weight_map = tf.tile(weight_map, [1, 1, 1, 12]) # BxHxWx12
    weighted_coefficient_map = coefficient_map * tile_weight_map
    coefficient_map_reshape = tf.reshape(weighted_coefficient_map, [spec.batch_size, -1, 12]) # BxHWx12
    square_coefficient = tf.matmul(coefficient_map_reshape, coefficient_map_reshape, transpose_a=True, name=name) # Bx12x12
    return square_coefficient

def read_lines(filepath):
    with open(filepath) as fin:
        lines = fin.readlines()
    lines = [line.strip() for line in lines]
    return lines

def preprocess_image(origin_image, spec):
    image = tf.cast(origin_image, tf.float32)
    image = tf.multiply(tf.subtract(image, spec.mean), spec.scale)
    return image

def postprocess_image(image, spec):
    post_image = tf.divide(image, spec.scale) + spec.mean
    post_image = post_image / 255.0
    return post_image

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
    # Rotation
    angle = tf.squeeze(tf.random_uniform([1], -30, 30, dtype=tf.float32))
    radian = angle * math.pi / 180.0
    image = tf.contrib.image.rotate(image, radian, interpolation='NEAREST')

    x1 = tf.squeeze(tf.random_uniform([1], 0, 0.2, dtype=tf.float32))
    y1 = tf.squeeze(tf.random_uniform([1], 0, 0.2, dtype=tf.float32))
    ratio = tf.squeeze(tf.random_uniform([1], 0.8, 1 - tf.maximum(x1, y1), dtype=tf.float32))
    x2 = x1 + ratio
    y2 = y1 + ratio
    boxes = [[y1, x1, y2, x2]]
    box_ind = [0]
    image = tf.expand_dims(image, axis=0)
    image = tf.image.crop_and_resize(image, boxes, box_ind, spec.crop_size)  # 1xHxWxC
    image = tf.squeeze(image, axis=0)  # HxWxC
    return image

def image_shrink_augmentation(image, spec):
    # Rotation
    angle = tf.squeeze(tf.random_uniform([1], -30, 30, dtype=tf.float32))
    radian = angle * math.pi / 180.0
    image = tf.contrib.image.rotate(image, radian, interpolation='NEAREST')

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

def random_crop_augmentation(image, coord_map, spec):
    """
    :param image: HxWx3
    :param coord_map: HxWx4
    :param image_pixel_map: HxWx2
    :param spec:
    :return:
    """
    image = tf.cast(image, tf.float32)
    image = tf.image.random_brightness(image, 20)
    image = tf.image.random_contrast(image, 0.8, 1.2)

    concat_data = tf.concat([image, coord_map], axis=-1)  # HxWx9
    concat_data_crop = image_augmentation(concat_data, spec)  # HxWx9

    image_crop = tf.slice(concat_data_crop, [0, 0, 0], [-1, -1, 3])  # HxWx3
    coord_map_crop = tf.slice(concat_data_crop, [0, 0, 3], [-1, -1, 3])  # HxWx3
    mask_map_crop = tf.slice(concat_data_crop, [0, 0, 6], [-1, -1, 1])  # HxWx1

    mask_map_crop = tf.cast((mask_map_crop >= 1.0), tf.float32)  # HxWx1

    coord_map_crop = tf.concat((coord_map_crop, mask_map_crop), axis=2) # HxWx4

    return image_crop, coord_map_crop

def get_training_data(image_list, label_list, spec):

    def _training_data_queue(indexes, image_paths, label_paths, spec):

        with tf.name_scope('data_queue'):
            with tf.device('/device:CPU:0'):
                index_queue = tf.train.string_input_producer(indexes, shuffle=True)
                index = tf.string_to_number(index_queue.dequeue(), out_type=tf.int32)
                image_path = tf.gather(image_paths, index)
                label_path = tf.gather(label_paths, index)
                image = tf.image.decode_png(tf.read_file(image_path), channels=3)
                image.set_shape((spec.image_size[0], spec.image_size[1], spec.channels))
                image = tf.cast(image, tf.float32)

                if FLAGS.scene == 'GreatCourt' or FLAGS.scene == 'KingsCollege' or \
                                FLAGS.scene == 'OldHospital' or FLAGS.scene == 'ShopFacade' or \
                                FLAGS.scene == 'StMarysChurch' or FLAGS.scene == 'Street' or \
                                FLAGS.scene == 'Street-east' or FLAGS.scene == 'Street-west' or \
                                FLAGS.scene == 'Street-south' or FLAGS.scene == 'Street-north1' or \
                                FLAGS.scene == 'Street-north2':
                    label_shape = [480, 852, 4]
                    coord_map = tf.reshape(tf.decode_raw(tf.read_file(label_path), tf.float32), label_shape)
                    coord_map = tf.image.crop_to_bounding_box(coord_map, 0, 2, 480, 848)
                else:
                    label_shape = [spec.image_size[0], spec.image_size[1], 4]
                    coord_map = tf.reshape(tf.decode_raw(tf.read_file(label_path), tf.float32), label_shape)

            with tf.device('/device:GPU:%d' % FLAGS.gpu):
                crop_images = []
                crop_coord_maps = []
                indexes = []
                for i in range(1):
                    crop_image, crop_coord_map = random_crop_augmentation(image, coord_map, spec)
                    crop_images.append(crop_image)
                    crop_coord_maps.append(crop_coord_map)
                    indexes.append(index)

                batch_crop_image = tf.stack(crop_images, 0)
                batch_crop_coord_map = tf.stack(crop_coord_maps, 0)
                batch_index = tf.stack(indexes)

            with tf.device('/device:CPU:0'):
                return tf.train.shuffle_batch(
                    [batch_crop_image, batch_crop_coord_map, batch_index],
                    batch_size=spec.batch_size,
                    capacity=spec.batch_size * 4,
                    min_after_dequeue=spec.batch_size * 2,
                    enqueue_many=True,
                    num_threads=40)

    image_paths = read_lines(image_list)
    label_paths = read_lines(label_list)

    indexes = get_indexes(len(image_paths), True)
    indexes = [str(i) for i in indexes]
    return _training_data_queue(indexes, image_paths, label_paths, spec)


def run_training(image_list, label_list, is_training=True):

    spec = helper.get_data_spec(model_class=ScoreNet)
    batch_images, batch_labels, batch_index = \
        get_training_data(image_list, label_list, spec)
    # BxHxWx3, BxHxWx4, BxHxWx2, Bx12

    with tf.device('/device:GPU:%d' % FLAGS.gpu):

        with tf.variable_scope("ScoreNet"):
            scorenet = ScoreNet({'input': batch_images},
                                 is_training=is_training,
                                 reuse=False)# BxHxWx4

        with tf.name_scope('loss'):
            gt_coords = tf.slice(batch_labels, [0, 0, 0, 0], [-1, -1, -1, 3], name='gt_coords') # BxHxWx3
            mask = tf.slice(batch_labels, [0, 0, 0, 3], [-1, -1, -1, 1], name='mask') # BxHxWx1
            # resize
            batch_images = tf.image.resize_bilinear(batch_images, [spec.crop_size[0] // spec.downsample, spec.crop_size[1] // spec.downsample])
            gt_coords = tf.image.resize_nearest_neighbor(gt_coords, [spec.crop_size[0] // spec.downsample, spec.crop_size[1] // spec.downsample])
            mask = tf.image.resize_nearest_neighbor(mask, [spec.crop_size[0] // spec.downsample, spec.crop_size[1] // spec.downsample])
            mask = tf.cast((mask >= 1.0), tf.float32)

            # coordinate loss
            coord_map, uncertainty_map = scorenet.GetOutput()
            transform = get_transform()
            gt_coords = ApplyTransform(gt_coords, transform, inverse=True)
            dist_threshold = 0.05

            if FLAGS.scene == 'ShopFacade' \
                    or FLAGS.scene == 'OldHospital':
                gt_coords *= 0.1
                dist_threshold = 0.01
            elif FLAGS.scene == 'StMarysChurch':
                gt_coords *= 0.05
                dist_threshold = 0.005
            elif FLAGS.scene == 'KingsCollege':
                gt_coords *= 0.02
                dist_threshold = 0.002
            elif FLAGS.scene == 'GreatCourt':
                gt_coords *= 0.01
                dist_threshold = 0.005
            elif FLAGS.scene == 'Street':
                gt_coords *= 0.005
                dist_threshold = 0.005
            elif FLAGS.scene == 'Street-east' or FLAGS.scene == 'Street-west' or FLAGS.scene == 'Street-south' \
                    or FLAGS.scene == 'Street-north1' or FLAGS.scene == 'Street-north2':
                gt_coords *= 0.02
                dist_threshold = 0.002
            elif FLAGS.scene == 'DeepLoc':
                gt_coords *= 0.02
                dist_threshold = 0.002

            coord_loss, accuracy = CoordLossWithUncertainty(coord_map, uncertainty_map, gt_coords, mask,
                                                            dist_threshold=dist_threshold)

            mean_coord_error, _ = CoordLoss(coord_map, gt_coords, mask, dist_threshold)

            # smooth loss
            smooth_loss = SmoothLoss(coord_map, batch_images, mask)

            weight1 = 1.0
            coord_loss = weight1 * coord_loss
            weight3 = 50.0
            smooth_loss = weight3 * smooth_loss
            loss = coord_loss #+ smooth_loss

            with tf.device('/device:CPU:0'):
                tf.summary.scalar('mean_coord_error(cm)', mean_coord_error * 100)
                tf.summary.scalar('accuracy', accuracy)

    return loss, coord_loss, smooth_loss, accuracy, batch_index


def get_testing_data(indexes, image_list, label_list, spec):

    def _testing_data_queue(indexes, image_paths, label_paths, spec):

        with tf.name_scope('data_queue'):
            index_queue = tf.train.string_input_producer(indexes, shuffle=FLAGS.shuffle)
            index = tf.string_to_number(index_queue.dequeue(), out_type=tf.int32)
            image_path = tf.gather(image_paths, index)
            label_path = tf.gather(label_paths, index)
            image = tf.image.decode_png(tf.read_file(image_path), channels=3)
            image.set_shape((spec.image_size[0], spec.image_size[1], spec.channels))
            image = tf.cast(image, tf.float32)
            if FLAGS.scene == 'GreatCourt' or FLAGS.scene == 'KingsCollege' or \
                            FLAGS.scene == 'OldHospital' or FLAGS.scene == 'ShopFacade' or \
                            FLAGS.scene == 'StMarysChurch' or FLAGS.scene == 'Street' or \
                            FLAGS.scene == 'Street-east' or FLAGS.scene == 'Street-west' or \
                            FLAGS.scene == 'Street-south' or FLAGS.scene == 'Street-north1' or \
                            FLAGS.scene == 'Street-north2':
                label_shape = [480, 852, 4]
                coord_map = tf.reshape(tf.decode_raw(tf.read_file(label_path), tf.float32), label_shape)
                coord_map = tf.image.crop_to_bounding_box(coord_map, 0, 2, 480, 848)
            else:
                label_shape = [spec.image_size[0], spec.image_size[1], 4]
                coord_map = tf.reshape(tf.decode_raw(tf.read_file(label_path), tf.float32), label_shape)

            # image, coord_map, pixel_map = random_crop_augmentation(image, coord_map, pixel_map, spec)
            # offset = 0
            # image = tf.image.crop_to_bounding_box(image, offset, offset, spec.crop_size[0], spec.crop_size[1])
            # coord_map = tf.image.crop_to_bounding_box(coord_map, offset, offset, spec.crop_size[0], spec.crop_size[1])
            # pixel_map = tf.image.crop_to_bounding_box(pixel_map, offset, offset, spec.crop_size[0], spec.crop_size[1])

            return tf.train.batch(
                [image, coord_map, index],
                batch_size=spec.batch_size,
                capacity=spec.batch_size * 2,
                num_threads=1)

    image_paths = read_lines(image_list)
    label_paths = read_lines(label_list)
    indexes = [str(i) for i in indexes]
    return _testing_data_queue(indexes, image_paths, label_paths, spec)

def run_testing(indexes, image_list, label_list, spec, is_training=False):

    batch_images, batch_labels, indexes = get_testing_data(indexes, image_list, label_list, spec)

    with tf.variable_scope("ScoreNet"):
        scorenet = ScoreNet({'input': batch_images},
                             is_training=is_training,
                             reuse=False)# BxHxWx4

    with tf.name_scope('loss'):
        gt_coords = tf.slice(batch_labels, [0, 0, 0, 0], [-1, -1, -1, 3], name='gt_coords')  # BxHxWx3
        mask = tf.slice(batch_labels, [0, 0, 0, 3], [-1, -1, -1, 1], name='mask')  # BxHxWx1
        gt_coords = tf.image.resize_nearest_neighbor(gt_coords, [spec.crop_size[0] // spec.downsample, spec.crop_size[1] // spec.downsample])
        mask = tf.image.resize_nearest_neighbor(mask, [spec.crop_size[0] // spec.downsample, spec.crop_size[1] // spec.downsample])
        mask = tf.cast((mask >= 1.0), tf.float32)

        # coordinate loss
        coord_map, uncertainty_map = scorenet.GetOutput()
        if FLAGS.scene == 'ShopFacade' \
                or FLAGS.scene == 'OldHospital':
            coord_map *= 10
            uncertainty_map *= 10
        elif FLAGS.scene == 'GreatCourt':
            coord_map *= 100
            uncertainty_map *= 100
        elif FLAGS.scene == 'StMarysChurch':
            coord_map *= 20
            uncertainty_map *= 20
        elif FLAGS.scene == 'KingsCollege':
            coord_map *= 50
            uncertainty_map *= 50
        elif FLAGS.scene == 'Street':
            coord_map *= 200
            uncertainty_map *= 200
        elif FLAGS.scene == 'Street-east' or FLAGS.scene == 'Street-west' or FLAGS.scene == 'Street-south' \
                or FLAGS.scene == 'Street-north1' or FLAGS.scene == 'Street-north2':
            coord_map *= 50
            uncertainty_map *= 50
        elif FLAGS.scene == 'DeepLoc':
            coord_map *= 50
            uncertainty_map *= 50

        transform = get_transform()
        coord_map = ApplyTransform(coord_map, transform)

    return scorenet, batch_images, coord_map, uncertainty_map, gt_coords, mask, indexes
