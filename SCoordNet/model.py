import math
from cnn_wrapper import helper, SCoordNet
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

def get_transform(transform_file = None):
    if transform_file:
        transform = np.loadtxt(transform_file, dtype=np.float32)
        transform = np.linalg.inv(transform)
    else:
        transform = np.array([[1.0, 0.0, 0.0, 0.0],
                              [0.0, 1.0, 0.0, 0.0],
                              [0.0, 0.0, 1.0, 0.0],
                              [0.0, 0.0, 0.0, 1.0]])
    return tf.constant(transform)

def get_indexes(num):
    return range(num)

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

    indexes = get_indexes(len(image_paths))
    indexes = [str(i) for i in indexes]
    return _training_data_queue(indexes, image_paths, label_paths, spec)


def run_training(image_list, label_list, transform_file, is_training=True):

    spec = helper.get_data_spec(model_class=SCoordNet)
    batch_images, batch_labels, batch_index = \
        get_training_data(image_list, label_list, spec)
    # BxHxWx3, BxHxWx4, BxHxWx2, Bx12

    with tf.device('/device:GPU:%d' % FLAGS.gpu):

        with tf.variable_scope("ScoreNet"):
            scoordnet = SCoordNet({'input': batch_images},
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
            coord_map, uncertainty_map = scoordnet.GetOutput()
            transform = get_transform(transform_file)
            gt_coords = ApplyTransform(gt_coords, transform, inverse=True)
            dist_threshold = 0.05

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

def run_testing(indexes, image_list, label_list, transform_file, spec, is_training=False):

    batch_images, batch_labels, indexes = get_testing_data(indexes, image_list, label_list, spec)

    with tf.variable_scope("ScoreNet"):
        scoordnet = SCoordNet({'input': batch_images},
                             is_training=is_training,
                             reuse=False)# BxHxWx4

    with tf.name_scope('loss'):
        gt_coords = tf.slice(batch_labels, [0, 0, 0, 0], [-1, -1, -1, 3], name='gt_coords')  # BxHxWx3
        mask = tf.slice(batch_labels, [0, 0, 0, 3], [-1, -1, -1, 1], name='mask')  # BxHxWx1
        gt_coords = tf.image.resize_nearest_neighbor(gt_coords, [spec.crop_size[0] // spec.downsample, spec.crop_size[1] // spec.downsample])
        mask = tf.image.resize_nearest_neighbor(mask, [spec.crop_size[0] // spec.downsample, spec.crop_size[1] // spec.downsample])
        mask = tf.cast((mask >= 1.0), tf.float32)

        # coordinate loss
        coord_map, uncertainty_map = scoordnet.GetOutput()
        transform = get_transform(transform_file)
        coord_map = ApplyTransform(coord_map, transform)

    return scoordnet, batch_images, coord_map, uncertainty_map, gt_coords, mask, indexes
