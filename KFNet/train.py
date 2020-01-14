import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from tools.io import read_lines
from KFNet import *
from tools.io import get_snapshot, get_num_trainable_params
import time
from tensorflow.python import debug as tf_debug
from util import *
from datetime import datetime

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
tf.app.flags.DEFINE_boolean('NIS', False, """Flag to NIS""")

# Params for solver.
tf.app.flags.DEFINE_float('base_lr', 0.0001,
                          """Base learning rate.""")
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
tf.app.flags.DEFINE_string('oflownet', '', """Path to OFlowNet model.""")
tf.app.flags.DEFINE_string('scoordnet', '', """Path to SCoordNet model.""")

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

def get_indexes(is_training):
    def _get_groups_in_range(start, end):
        groups = []
        for i in range(start, end - 3):
            groups.append([i, i + 1, i + 2, i + 3])
            groups.append([i + 3, i + 2, i + 1, i])
        return groups
    def _get_full_groups_in_range(start, end):
        groups = [[start+1, start]]
        for i in range(start, end - 1):
            groups.append([i, i + 1])
        return groups
    groups = []
    blind_groups = []

    if FLAGS.scene == 'chess':
        if is_training:
            groups += _get_groups_in_range(0, 1000)
            groups += _get_groups_in_range(1000, 2000)
            groups += _get_groups_in_range(2000, 3000)
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

def get_training_data(image_list, label_list, spec, is_training):
    image_paths = read_lines(image_list)
    label_paths = read_lines(label_list)

    groups = get_indexes(is_training)
    indexes = [str(i) for i in range(len(groups))]
    groups = tf.stack(groups, axis=0, name='group_indexes')

    with tf.name_scope('data_queue'):
        with tf.device('/CPU:0'):
            index_queue = tf.train.string_input_producer(indexes, shuffle=FLAGS.shuffle)
            index = tf.string_to_number(index_queue.dequeue(), out_type=tf.int32)
            group = tf.squeeze(tf.slice(groups, [index, 0], [1, -1]), name='group') # B
            group_image_paths = tf.gather(image_paths, group)
            group_label_paths = tf.gather(label_paths, group)

            images = []
            labels = []
            for i in range(spec.batch_size):
                image_path = tf.squeeze(tf.slice(group_image_paths, [i], [1]))
                image = tf.image.decode_png(tf.read_file(image_path), channels=spec.channels)
                image.set_shape((spec.image_size[0], spec.image_size[1], spec.channels))
                image = tf.cast(image, tf.float32)

                label_path = tf.squeeze(tf.slice(group_label_paths, [i], [1]))
                label_shape = [spec.image_size[0], spec.image_size[1], 4]
                label = tf.decode_raw(tf.read_file(label_path), tf.float32)
                label = tf.reshape(label, label_shape)

                images.append(image)
                labels.append(label)

            images = tf.stack(images, axis=0)   # BxHxWx3
            labels = tf.stack(labels, axis=0)  # BxHxWx4

            coords = tf.slice(labels, [0, 0, 0, 0], [-1, -1, -1, 3])
            masks = tf.slice(labels, [0, 0, 0, 3], [-1, -1, -1, 1])

        with tf.device('/CPU:0'):
            return tf.train.batch(
                [images, coords, masks, group],
                batch_size=spec.batch_size,
                capacity=spec.batch_size * 4,
                enqueue_many=True,
                num_threads=1)

def KF_fusion(image_list, label_list, transform, last_coord, last_uncertainty, spec):
    images, gt_coords, masks, group_indexes = \
        get_training_data(image_list, label_list, spec, False)

    sfmnet = KFNet(images, spec, train_scoordnet=False, train_oflownet=False)

    measure_coord, measure_uncertainty = sfmnet.GetMeasureCoord2()
    temp_coord, temp_uncertainty, KF_coord, KF_uncertainty = \
        sfmnet.GetKFCoordRecursive(last_coord, last_uncertainty)
    NIS = sfmnet.GetNIS(measure_coord, measure_uncertainty, temp_coord, temp_uncertainty)   # 1xHxWx3

    measure_loss, measure_accuracy = sfmnet.CoordLossWithUncertainty(measure_coord, measure_uncertainty, gt_coords,
                                                                     mask=masks, transform=transform, downsample=True)
    temp_loss, temp_accuracy = sfmnet.CoordLossWithUncertainty(temp_coord, temp_uncertainty, gt_coords,
                                                               mask=masks, transform=transform, downsample=True)
    KF_loss, KF_accuracy = sfmnet.CoordLossWithUncertainty(KF_coord, KF_uncertainty, gt_coords,
                                                           mask=masks, transform=transform, downsample=True)

    resize = [spec.image_size[0] / 8, spec.image_size[1] / 8]
    gt_coords_8 = tf.image.resize_nearest_neighbor(gt_coords, resize)
    masks_8 = tf.image.resize_nearest_neighbor(masks, resize)

    return measure_coord, measure_uncertainty,\
           temp_coord, temp_uncertainty, KF_coord, KF_uncertainty, NIS, group_indexes, \
           measure_loss, measure_accuracy, temp_loss, temp_accuracy, KF_loss, KF_accuracy, \
           gt_coords_8, masks_8

def run(image_list, label_list, transform_file, spec, is_training=True):
    with tf.name_scope('data'):
        images, gt_coords, masks, group_indexes = \
            get_training_data(image_list, label_list, spec, is_training)

    with tf.device('/device:CPU:0'):
        resize_images = tf.image.resize_nearest_neighbor(images, [spec.image_size[0] // 8, spec.image_size[1] // 8])
        gt_coords = tf.image.resize_nearest_neighbor(gt_coords, [spec.image_size[0] // 8, spec.image_size[1] // 8])
        masks = tf.image.resize_nearest_neighbor(masks, [spec.image_size[0] // 8, spec.image_size[1] // 8])
        masks = tf.cast(tf.equal(masks, 1.0), tf.float32)

    transform = get_transform(transform_file)
    gt_coords = ApplyTransform(gt_coords, transform, inverse=True)

    kfnet = KFNet(images, spec, train_scoordnet=is_training,
                  train_oflownet=is_training and not FLAGS.fix_flownet,
                  reuse=tf.AUTO_REUSE)

    with tf.name_scope('loss'):
        measure_coord_loss, measure_coord_accuracy = kfnet.MeasureCoordLoss(gt_coords, masks,
                                                                             images=resize_images)
        temp_coord_loss, temp_coord_accuracy = kfnet.TemporalCoordLoss(gt_coords, masks,
                                                                        images=resize_images)
        KF_coord_loss, KF_coord_accuracy = kfnet.KFCoordLoss(gt_coords, masks,
                                                              images=resize_images)
        loss = 0.2 * measure_coord_loss \
               + 0.2 * temp_coord_loss \
               + 0.6 * KF_coord_loss

    return kfnet, loss, measure_coord_loss, measure_coord_accuracy, temp_coord_loss, temp_coord_accuracy, \
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
    if FLAGS.scoordnet != '' and FLAGS.oflownet != '':
        print 'Reset step to zero for retraining.'
        FLAGS.reset_step = FLAGS.stepvalue * 4
    FLAGS.max_steps = FLAGS.stepvalue * 5

def train(image_list, label_list, transform_file, out_dir, \
          snapshot=None, step=0, debug=False, gpu=0):

    print image_list
    image_paths = read_lines(image_list)
    spec = KFNetDataSpec()
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
            = run(image_list, label_list, transform_file, spec)
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

        if FLAGS.scoordnet != '':
            snapshot, _ = get_snapshot(FLAGS.scoordnet)
            RestoreFromScope(sess, snapshot, 'ScoreNet')
        if FLAGS.oflownet != '':
            snapshot, _ = get_snapshot(FLAGS.oflownet)
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
                format_str = '[%s] epoch %d, step %d/%d, %5d~%5d~%5d~%5d, loss=%.3f, l_measure=%.3f, l_temp=%.3f, l_KF= %.3f, ' \
                             'a_measure=%.3f, a_temp=%.3f,a_KF=%.3f, #pixels=%d, lr = %.6f (%.3f sec/step)'
                print(format_str % (datetime.now(), epoch, step, FLAGS.max_steps,
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
    transform_file = os.path.join(FLAGS.input_folder, 'transform.txt')

    train(image_list, label_list, transform_file, FLAGS.model_folder,
          snapshot, step, FLAGS.debug, FLAGS.gpu)


if __name__ == '__main__':
    tf.app.run()
