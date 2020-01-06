import sys
sys.path.append('/run/media/larry/fafb882a-0878-4e0a-9ccb-2fb979b7f717/e3dengine/tfmatch')
import tensorflow as tf
from tools.io import read_lines
import numpy as np
from SfMNet import *
from tools.common import Notify
from tools.io import get_snapshot, get_num_trainable_params
import time, os, argparse
from tensorflow.python import debug as tf_debug
from util import *

FLAGS = tf.app.flags.FLAGS

# Params for config.
tf.app.flags.DEFINE_boolean('is_training', True, """Flag to training model""")
tf.app.flags.DEFINE_boolean('debug', False, """Flag to debug""")
tf.app.flags.DEFINE_boolean('show', False, """Flag to visualize""")
tf.app.flags.DEFINE_boolean('shuffle', False, """Flag to shuffle""")
tf.app.flags.DEFINE_boolean('fix_flownet', False, """Flag to fix flownet""")
tf.app.flags.DEFINE_integer('gpu', 0, """GPU id.""")
tf.app.flags.DEFINE_string('scene', '', """Path to save the model.""")
tf.app.flags.DEFINE_integer('reset_step', -1, """Reset training step.""")

# Params for solver.
tf.app.flags.DEFINE_float('base_lr', 0.0001,
                          """Base learning rate.""")
# tf.app.flags.DEFINE_float('base_lr', 0.0001,
#                           """Base learning rate.""")
tf.app.flags.DEFINE_integer('max_steps', 400000,
                            """Max training iteration.""")
tf.app.flags.DEFINE_integer('display', 10,
                            """Interval of loginfo display.""")
tf.app.flags.DEFINE_integer('stepvalue', 80000,
                            """Step interval to decay learning rate.""")
tf.app.flags.DEFINE_integer('snapshot', 5000,
                            """Step interval to save the model.""")
tf.app.flags.DEFINE_float('gamma', 0.5,
                          """Learning rate decay rate.""")
tf.app.flags.DEFINE_float('weight_decay', 0.0001,
                          """Fraction of regularization term.""")
tf.app.flags.DEFINE_float('momentum', 0.9,
                          """momentum in when using SGD with momentum solver""")

# Params for I/O
tf.app.flags.DEFINE_string('input_folder', '', """Path to data.""")
tf.app.flags.DEFINE_string('model_folder', '', """Path to model.""")
tf.app.flags.DEFINE_string('output_folder', '', """Path to output.""")
tf.app.flags.DEFINE_string('flownet', '', """Path to flownet model.""")
tf.app.flags.DEFINE_string('scorenet', '', """Path to scorenet model.""")

def get_7scene_transform():
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
    else:
        print 'Invalid scene:', FLAGS.scene
        exit()
    return transform

def get_indexes(is_training):
    def _get_groups_in_range(start, end):
        groups = []
        for i in range(start, end - 3):
            groups.append([i, i + 1, i + 2, i + 3])
            groups.append([i + 3, i + 2, i + 1, i])
        return groups
    def _get_full_groups_in_range(start, end):
        groups = [[start+3, start+2, start+1, start],
                  [start+4, start+3, start+2, start+1],
                  [start+5, start+4, start+3, start+2]]
        for i in range(start, end - 3):
            groups.append([i, i + 1, i + 2, i + 3])
        return groups
    def _groups_from_pair(pairs):
        groups = []
        for pair in pairs:
            groups.append([pair[0], pair[1], pair[1]+1, pair[1]+2])
            groups.append([pair[0] - 1, pair[0], pair[1], pair[1]+1])
            groups.append([pair[0] - 2, pair[0] - 1, pair[0], pair[1]])
            groups.append([pair[1] + 2, pair[1] + 1, pair[1], pair[0]])
            groups.append([pair[1] + 1, pair[1], pair[0], pair[0] - 1])
            groups.append([pair[1], pair[0], pair[0] - 1, pair[0] - 2])
        return groups

    groups = []
    blind_groups = []

    if FLAGS.scene == 'chess':
        if is_training:
            groups += _get_groups_in_range(0, 1000)
            groups += _get_groups_in_range(1000, 2000)
            groups += _get_groups_in_range(2000, 2924)
            groups += _get_groups_in_range(2944, 3000)
            groups += _get_groups_in_range(3000, 4000)
        else:
            groups += _get_full_groups_in_range(0, 1000)
            groups += _get_full_groups_in_range(1000, 2000)
    elif FLAGS.scene == 'fire':
        if is_training:
            groups += _get_groups_in_range(0, 1000)
            groups += _get_groups_in_range(1000, 2000)
        else:
            groups += _get_full_groups_in_range(0, 1000)
            groups += _get_full_groups_in_range(1000, 2000)
    elif FLAGS.scene == 'heads':
        if is_training:
            groups += _get_groups_in_range(0, 1000)
        else:
            groups += _get_full_groups_in_range(0, 1000)
    elif FLAGS.scene == 'office':
        if is_training:
            groups += _get_groups_in_range(0, 1000)
            groups += _get_groups_in_range(1000, 2000)
            groups += _get_groups_in_range(2000, 3000)
            groups += _get_groups_in_range(3000, 4000)
            groups += _get_groups_in_range(4000, 5000)
            groups += _get_groups_in_range(5000, 6000)
            pairs = [[236, 237], [237, 238], [241, 242], [242, 243], [246, 247], [255, 256], [256, 257], [258, 259], [259, 260], [385, 386], [386, 387], [400, 401], [413, 414], [414, 415], [601, 602], [602, 603], [622, 623], [624, 625], [651, 652], [723, 724], [731, 732], [787, 788], [788, 789], [861, 862], [862, 863], [869, 870], [870, 871], [875, 876], [987, 988], [988, 989], [1173, 1174], [1174, 1175], [1276, 1277], [1283, 1284], [1293, 1294], [1492, 1493], [1544, 1545], [1545, 1546], [1662, 1663], [1663, 1664], [1673, 1674], [1674, 1675], [1688, 1689], [1689, 1690], [1700, 1701], [1747, 1748], [1748, 1749], [1851, 1852], [2238, 2239], [2239, 2240], [2251, 2252], [2359, 2360], [2779, 2780], [3022, 3023], [3023, 3024], [3058, 3059], [3143, 3144], [3261, 3262], [3307, 3308], [3315, 3316], [3318, 3319], [3321, 3322], [3661, 3662], [3672, 3673], [3674, 3675], [3709, 3710], [3867, 3868], [3877, 3878], [3878, 3879], [3884, 3885], [4073, 4074], [4074, 4075], [4093, 4094], [4099, 4100], [4104, 4105], [4194, 4195], [4195, 4196], [4199, 4200], [4200, 4201], [4284, 4285], [4285, 4286], [4339, 4340], [4340, 4341], [4458, 4459], [4462, 4463], [4484, 4485], [4495, 4496], [4589, 4590], [4652, 4653], [4653, 4654], [4717, 4718], [4718, 4719], [4743, 4744], [4750, 4751], [4751, 4752], [4754, 4755], [4756, 4757], [4757, 4758], [4758, 4759], [4759, 4760], [4760, 4761], [4761, 4762], [4766, 4767], [4767, 4768], [4944, 4945], [4951, 4952], [4952, 4953], [4980, 4981], [5129, 5130], [5136, 5137], [5153, 5154], [5238, 5239], [5239, 5240], [5259, 5260], [5260, 5261], [5269, 5270], [5270, 5271], [5275, 5276], [5276, 5277], [5319, 5320], [5433, 5434], [5434, 5435], [5604, 5605], [5642, 5643], [5643, 5644], [5646, 5647], [5647, 5648], [5650, 5651], [5651, 5652], [5660, 5661], [5661, 5662], [5673, 5674], [5674, 5675], [5742, 5743], [5774, 5775], [5776, 5777], [5829, 5830], [5876, 5877], [5877, 5878], [5979, 5980]]
            blind_groups = _groups_from_pair(pairs)
        else:
            groups += _get_full_groups_in_range(0, 1000)
            groups += _get_full_groups_in_range(1000, 2000)
            groups += _get_full_groups_in_range(2000, 3000)
            groups += _get_full_groups_in_range(3000, 4000)
    elif FLAGS.scene == 'pumpkin':
        if is_training:
            groups += _get_groups_in_range(0, 1000)
            groups += _get_groups_in_range(1000, 2000)
            groups += _get_groups_in_range(2000, 3000)
            groups += _get_groups_in_range(3000, 4000)
            pairs = [[25, 26], [131, 132], [132, 133], [145, 146], [156, 157], [244, 245], [245, 246], [250, 251], [265, 266], [266, 267], [270, 271], [276, 277], [277, 278], [316, 317], [317, 318], [318, 319], [320, 321], [361, 362], [362, 363], [484, 485], [524, 525], [525, 526], [532, 533], [555, 556], [1130, 1131], [1188, 1189], [1189, 1190], [1199, 1200], [1200, 1201], [1229, 1230], [1352, 1353], [1896, 1897], [1897, 1898], [2248, 2249], [2264, 2265], [2266, 2267], [2283, 2284], [2284, 2285], [2292, 2293], [2293, 2294], [2447, 2448], [2462, 2463], [2463, 2464], [2470, 2471], [2859, 2860], [2860, 2861], [2871, 2872], [2872, 2873], [2879, 2880], [2926, 2927], [3084, 3085], [3085, 3086], [3087, 3088], [3103, 3104], [3318, 3319], [3337, 3338], [3338, 3339], [3499, 3500], [3511, 3512], [3553, 3554], [3557, 3558], [3735, 3736], [3736, 3737], [3738, 3739], [3746, 3747], [3747, 3748], [3751, 3752], [3752, 3753], [3814, 3815], [3815, 3816], [3816, 3817], [3820, 3821], [3842, 3843], [3845, 3846], [3846, 3847], [3958, 3959], [3960, 3961], [3961, 3962], [3964, 3965], [3965, 3966], [3975, 3976], [3976, 3977], [3983, 3984], [3985, 3986], [3990, 3991]]
            blind_groups = _groups_from_pair(pairs)
        else:
            groups += _get_full_groups_in_range(0, 1000)
            groups += _get_full_groups_in_range(1000, 2000)
    elif FLAGS.scene == 'redkitchen':
        if is_training:
            groups += _get_groups_in_range(0, 1000)
            groups += _get_groups_in_range(1000, 2000)
            groups += _get_groups_in_range(2000, 3000)
            groups += _get_groups_in_range(3000, 4000)
            groups += _get_groups_in_range(4000, 5000)
            groups += _get_groups_in_range(5000, 6000)
            groups += _get_groups_in_range(6000, 7000)
            pairs = [[121, 122], [178, 179], [312, 313], [340, 341], [345, 346], [346, 347], [447, 448], [453, 454], [733, 734], [917, 918], [1055, 1056], [1520, 1521], [1626, 1627], [1627, 1628], [1711, 1712], [1718, 1719], [1719, 1720], [1856, 1857], [1945, 1946], [1946, 1947], [2142, 2143], [2143, 2144], [2212, 2213], [2213, 2214], [2228, 2229], [2287, 2288], [2288, 2289], [2317, 2318], [2318, 2319], [2411, 2412], [2412, 2413], [2419, 2420], [2420, 2421], [2436, 2437], [2497, 2498], [2498, 2499], [2574, 2575], [2581, 2582], [2582, 2583], [2623, 2624], [2722, 2723], [2898, 2899], [2973, 2974], [2974, 2975], [3191, 3192], [3192, 3193], [3215, 3216], [3216, 3217], [3221, 3222], [3222, 3223], [3303, 3304], [3461, 3462], [3485, 3486], [3511, 3512], [3513, 3514], [4099, 4100], [4100, 4101], [4125, 4126], [4126, 4127], [4130, 4131], [4131, 4132], [4135, 4136], [4136, 4137], [4155, 4156], [4156, 4157], [4208, 4209], [4209, 4210], [4359, 4360], [4360, 4361], [4427, 4428], [4504, 4505], [4505, 4506], [4550, 4551], [4551, 4552], [4552, 4553], [4706, 4707], [4726, 4727], [5092, 5093], [5093, 5094], [5163, 5164], [5219, 5220], [5231, 5232], [5232, 5233], [5356, 5357], [5357, 5358], [5362, 5363], [5419, 5420], [5420, 5421], [5444, 5445], [5783, 5784], [5784, 5785], [5850, 5851], [6051, 6052], [6052, 6053], [6318, 6319], [6333, 6334], [6334, 6335], [6337, 6338], [6338, 6339], [6399, 6400], [6503, 6504], [6504, 6505], [6512, 6513], [6513, 6514], [6542, 6543], [6543, 6544], [6590, 6591], [6591, 6592], [6778, 6779], [6779, 6780]]
            blind_groups = _groups_from_pair(pairs)
        else:
            groups += _get_full_groups_in_range(0, 1000)
            groups += _get_full_groups_in_range(1000, 2000)
            groups += _get_full_groups_in_range(2000, 3000)
            groups += _get_full_groups_in_range(3000, 4000)
            groups += _get_full_groups_in_range(4000, 5000)
    elif FLAGS.scene == 'stairs':
        if is_training:
            groups += _get_groups_in_range(0, 500)
            groups += _get_groups_in_range(500, 1000)
            groups += _get_groups_in_range(1000, 1500)
            groups += _get_groups_in_range(1500, 2000)
        else:
            groups += _get_full_groups_in_range(0, 500)
            groups += _get_full_groups_in_range(500, 1000)
    else:
        print 'Invalid scene:', FLAGS.scene
        exit()
    groups = [i for i in groups if i not in blind_groups]
    print groups
    return groups


def get_pixel_map(height, width, spec):
    """
    Get the unnormalized pixel coordinates
    :return: HxWx2
    """
    map_y = np.zeros((height, width))  # HxW
    map_x = np.zeros((height, width))  # HxW
    for i in range(height):
        map_x[i, :] = range(width)
    for i in range(width):
        map_y[:, i] = range(height)
    map_y = tf.convert_to_tensor(map_y, tf.float32)
    map_y = (map_y - spec.v) / spec.focal_y
    map_x = tf.convert_to_tensor(map_x, tf.float32)
    map_x = (map_x - spec.u) / spec.focal_x
    map = tf.stack([map_x, map_y], axis=-1) # HxWx2
    return map

def data_augmentation(image, coord_map, image_pixel_map, spec):
    """
    :param image: BxHxWx3
    :param coord_map: BxHxWx4
    :param image_pixel_map: BxHxWx2
    :param spec:
    :return:
    """
    # color augmentation
    image = tf.cast(image, tf.float32)
    image = tf.image.random_brightness(image, 20)
    image = tf.image.random_contrast(image, 0.8, 1.2)

    concat_data = tf.concat([image, coord_map, image_pixel_map], axis=-1)  # BxHxWx9
    concat_data = image_augmentation(concat_data, spec)  # BxHxWx9

    image_crop = tf.slice(concat_data, [0, 0, 0, 0], [-1, -1, -1, 3])  # BxHxWx3
    coord_map_crop = tf.slice(concat_data, [0, 0, 0, 3], [-1, -1, -1, 3])  # BxHxWx3
    mask_map_crop = tf.slice(concat_data, [0, 0, 0, 6], [-1, -1, -1, 1])  # BxHxWx1
    pixel_map_crop = tf.slice(concat_data, [0, 0, 0, 7], [-1, -1, -1, 2])  # BxHxWx2

    mask_map_crop = tf.cast((mask_map_crop >= 1.0), tf.float32)  # BxHxWx1

    coord_map_crop = tf.concat((coord_map_crop, mask_map_crop), axis=-1) # BxHxWx4

    return image_crop, coord_map_crop, pixel_map_crop

def get_training_data(image_list, label_list, pose_file, spec, is_training=True):
    image_paths = read_lines(image_list)
    label_paths = read_lines(label_list)
    with tf.device('/CPU:0'):
        poses = tf.decode_raw(tf.read_file(pose_file), tf.float32)
    poses = tf.reshape(poses, [-1, 4, 4], name='poses')

    groups = get_indexes(is_training)
    indexes = [str(i) for i in range(len(groups))]
    groups = tf.stack(groups, axis=0, name='group_indexes')

    with tf.name_scope('data_queue'):
        with tf.device('/CPU:0'):
            index_queue = tf.train.string_input_producer(indexes, shuffle=is_training or FLAGS.shuffle)
            index = tf.string_to_number(index_queue.dequeue(), out_type=tf.int32)
            group = tf.squeeze(tf.slice(groups, [index, 0], [1, -1]), name='group') # B
            group_image_paths = tf.gather(image_paths, group)
            group_label_paths = tf.gather(label_paths, group)

            images = []
            labels = []
            pixel_maps = []
            local_poses = []
            for i in range(spec.batch_size):
                image_path = tf.squeeze(tf.slice(group_image_paths, [i], [1]))
                image = tf.image.decode_png(tf.read_file(image_path), channels=spec.channels)
                image.set_shape((spec.image_size[0], spec.image_size[1], spec.channels))
                image = tf.cast(image, tf.float32)

                label_path = tf.squeeze(tf.slice(group_label_paths, [i], [1]))
                label_shape = [spec.image_size[0], spec.image_size[1], 4]
                label = tf.decode_raw(tf.read_file(label_path), tf.float32)
                label = tf.reshape(label, label_shape)

                pixel_map = get_pixel_map(spec.image_size[0], spec.image_size[1], spec)

                local_pose = tf.squeeze(tf.slice(poses, [group[i], 0, 0], [1, -1, -1]))

                images.append(image)
                labels.append(label)
                pixel_maps.append(pixel_map)
                local_poses.append(local_pose)

            images = tf.stack(images, axis=0)   # BxHxWx3
            labels = tf.stack(labels, axis=0)  # BxHxWx4
            pixel_maps = tf.stack(pixel_maps, axis=0)   # BxHxWx2
            local_poses = tf.stack(local_poses, axis=0)  # Bx4x4

            if is_training:
                images, labels, pixel_maps = data_augmentation(images, labels, pixel_maps, spec)

            coords = tf.slice(labels, [0, 0, 0, 0], [-1, -1, -1, 3])
            masks = tf.slice(labels, [0, 0, 0, 3], [-1, -1, -1, 1])

        if is_training:
            num_threads = 40
        else:
            num_threads = 1
        with tf.device('/CPU:0'):
            return tf.train.batch(
                [images, coords, masks, pixel_maps, local_poses, group],
                batch_size=spec.batch_size,
                capacity=spec.batch_size * 4,
                enqueue_many=True,
                num_threads=num_threads)

def run(image_list, label_list, pose_file, spec, is_training=True):
    with tf.name_scope('data'):
        images, gt_coords, masks, pixel_maps, poses, group_indexes = \
            get_training_data(image_list, label_list, pose_file, spec, is_training)

    with tf.device('/device:CPU:0'):
        resize_images = tf.image.resize_nearest_neighbor(images, [spec.image_size[0] // 8, spec.image_size[1] // 8])
        gt_coords = tf.image.resize_nearest_neighbor(gt_coords, [spec.image_size[0] // 8, spec.image_size[1] // 8])
        masks = tf.image.resize_nearest_neighbor(masks, [spec.image_size[0] // 8, spec.image_size[1] // 8])
        masks = tf.cast(tf.equal(masks, 1.0), tf.float32)
        pixel_maps = tf.image.resize_nearest_neighbor(pixel_maps, [spec.image_size[0] // 8, spec.image_size[1] // 8])

    transform = get_7scene_transform()
    gt_coords = ApplyTransform(gt_coords, transform, inverse=True)

    sfmnet = SfMNet(images, gt_coords, spec.focal_x, spec.focal_y, spec.u, spec.v,
                    train_scorenet=is_training, train_temporal=is_training and not FLAGS.fix_flownet,
                    reuse=tf.AUTO_REUSE)

    with tf.name_scope('loss'):
        measure_coord_loss, measure_coord_accuracy = sfmnet.MeasureCoordLoss(gt_coords, masks,
                                                                             images=resize_images)
        temp_coord_loss, temp_coord_accuracy = sfmnet.TemporalCoordLoss(gt_coords, masks,
                                                                        images=resize_images)
        KF_coord_loss, KF_coord_accuracy = sfmnet.KFCoordLoss(gt_coords, masks,
                                                              images=resize_images)
        loss = 0.2 * measure_coord_loss \
               + 0.2 * temp_coord_loss \
               + 0.6 * KF_coord_loss

    return sfmnet, loss, measure_coord_loss, measure_coord_accuracy, temp_coord_loss, temp_coord_accuracy, \
           KF_coord_loss, KF_coord_accuracy, group_indexes, gt_coords * masks, masks

def solver(loss):
    weights_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    reg_loss = tf.contrib.layers.apply_regularization(
        tf.contrib.layers.l2_regularizer(scale=FLAGS.weight_decay), weights_list)
    with tf.device('/CPU:0'):
        global_step = tf.Variable(initial_value=0, trainable=False, name='global_step')
        lr_op = tf.train.exponential_decay(FLAGS.base_lr,
                                           global_step=global_step,
                                           decay_steps=FLAGS.stepvalue,
                                           decay_rate=FLAGS.gamma,
                                           name='lr')
    bn_list = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(bn_list):
        opt = tf.train.AdamOptimizer(learning_rate=lr_op).minimize(
            loss + reg_loss, global_step=global_step)
    return opt, lr_op, reg_loss

def RestoreFromScope(sess, snapshot, scope):
    print 'Restore from scope', scope, ':', snapshot
    variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)
    restorer = tf.train.Saver(variables)
    restorer.restore(sess, snapshot)

def RestoreVariableByName(sess, snapshot, name):
    print 'Restore variable:', name
    variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    restore_variables = [v for v in variables if v.name == name]
    restorer = tf.train.Saver(restore_variables)
    restorer.restore(sess, snapshot)

def set_stepvalue():
    if FLAGS.scene == 'chess':
        FLAGS.stepvalue = 100000
    elif FLAGS.scene == 'fire':
        FLAGS.stepvalue = 30000
    elif FLAGS.scene == 'heads':
        FLAGS.stepvalue = 60000
    elif FLAGS.scene == 'office':
        FLAGS.stepvalue = 140000
    elif FLAGS.scene == 'pumpkin':
        FLAGS.stepvalue = 140000
    elif FLAGS.scene == 'redkitchen':
        FLAGS.stepvalue = 140000
    elif FLAGS.scene == 'stairs':
        FLAGS.stepvalue = 100000
    else:
        print 'Invalid scene:', FLAGS.scene
        exit()
    if FLAGS.scorenet != '' and FLAGS.flownet != '':
        print 'Reset step to zero for retraining.'
        FLAGS.reset_step = FLAGS.stepvalue * 4
    FLAGS.max_steps = FLAGS.stepvalue * 5

def train(image_list, label_list, pose_file, out_dir, \
          snapshot=None, step=0, debug=False, gpu=0):

    print image_list
    image_paths = read_lines(image_list)
    spec = SfMNetDataSpec()
    spec.scene = FLAGS.scene
    spec.image_num = len(image_paths)
    spec.sequence_length = 500
    spec.num_sequence = spec.image_num // spec.sequence_length
    set_stepvalue()
    if FLAGS.reset_step >= 0:
        step = FLAGS.reset_step
    print "----------------------------------"
    print "scene: ", spec.scene
    print "training image number: ", len(image_paths)
    print "batch size: ", spec.batch_size
    print "step value: ", FLAGS.stepvalue
    print "max steps: ", FLAGS.max_steps
    print "current step: ", step
    print "----------------------------------"

    with tf.device('/device:GPU:%d' % gpu):
        _, loss, measure_loss, measure_accuracy, temp_loss, temp_accuracy, KF_loss, KF_accuracy, group_indexes, _, masks \
            = run(image_list, label_list, pose_file, spec)
        optimizer, lr_op, reg_loss = solver(loss)
        init_op = tf.global_variables_initializer()
        global_step = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="global_step")[0]
        init_step = global_step.assign(step)

    print "----------------------------------"
    print "# trainable params: ", get_num_trainable_params()
    print "----------------------------------"
    raw_input("Please check the meta info, press any key to continue...")

    # configuration
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=None)
    summary_op = tf.summary.merge_all()

    config = tf.ConfigProto(log_device_placement=False)
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        if debug:
            sess = tf_debug.LocalCLIDebugWrapperSession(sess)

        if snapshot:
            RestoreFromScope(sess, snapshot, None)
        else:
            sess.run(init_op)

        if FLAGS.scorenet != '':
            snapshot, _ = get_snapshot(FLAGS.scorenet)
            RestoreFromScope(sess, snapshot, 'ScoreNet')
        if FLAGS.flownet != '':
            snapshot, _ = get_snapshot(FLAGS.flownet)
            RestoreFromScope(sess, snapshot, 'Temporal')

        sess.run(init_step)

        # Start populating the queue.
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        while step <= FLAGS.max_steps:
            start_time = time.time()
            summary_str, _, lr, out_loss, out_measure_loss, out_measure_accuracy, out_temp_loss, out_temp_accuracy, \
            out_KF_loss, out_KF_accuracy, out_group_indexes, out_masks = \
                sess.run([summary_op, optimizer, lr_op, loss, measure_loss, measure_accuracy, temp_loss, temp_accuracy,
                          KF_loss, KF_accuracy, group_indexes, masks])
            duration = time.time() - start_time

            # Print info.
            if step % FLAGS.display == 0 or not FLAGS.is_training:
                epoch = step // spec.image_num
                format_str = 'epoch %d, step %d/%d, %5d~%5d~%5d~%5d, loss=%.3f, l_measure=%.3f, l_temp=%.3f, l_KF= %.3f, ' \
                             'a_measure=%.3f, a_temp=%.3f,a_KF=%.3f, #pixels=%d, lr = %.6f (%.3f sec/step)'
                print(Notify.INFO, format_str % (epoch, step, FLAGS.max_steps,
                                                 out_group_indexes[0], out_group_indexes[1], out_group_indexes[2], out_group_indexes[3],
                                                 out_loss, out_measure_loss, out_temp_loss, out_KF_loss, out_measure_accuracy,
                                                 out_temp_accuracy, out_KF_accuracy, np.sum(out_masks), lr, duration))

            # Save the model checkpoint periodically.
            if step % FLAGS.snapshot == 0 or step == FLAGS.max_steps:
                checkpoint_path = os.path.join(out_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)
            step += 1

        coord.request_stop()
        coord.join(threads)

def main(_):
    snapshot, step = get_snapshot(FLAGS.model_folder)

    image_list = os.path.join(FLAGS.input_folder, 'image_list.txt')
    label_list = os.path.join(FLAGS.input_folder, 'label_list.txt')
    pose_file = os.path.join(FLAGS.input_folder, 'poses.bin')

    train(image_list, label_list, pose_file, FLAGS.model_folder,
          snapshot, step, FLAGS.debug, FLAGS.gpu)


if __name__ == '__main__':
    tf.app.run()
