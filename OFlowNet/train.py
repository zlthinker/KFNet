import sys, os, time
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from tools.io import read_lines
from KFNet import *
from tools.io import get_snapshot, get_num_trainable_params
from tensorflow.python import debug as tf_debug
from util import *
from datetime import datetime

FLAGS = tf.app.flags.FLAGS

# Params for config.
tf.app.flags.DEFINE_boolean('is_training', True, """Flag to training model""")
tf.app.flags.DEFINE_boolean('debug', False, """Flag to debug""")
tf.app.flags.DEFINE_boolean('show', False, """Flag to visualize""")
tf.app.flags.DEFINE_integer('gpu', 0, """GPU id.""")
tf.app.flags.DEFINE_string('scene', '', """Path to save the model.""")
tf.app.flags.DEFINE_boolean('noise', False, """Flag to add noise to gt_coords""")
tf.app.flags.DEFINE_boolean('shuffle', False, """Flag to shuffle data""")

# Params for solver.
tf.app.flags.DEFINE_float('base_lr', 0.0001,
                          """Base learning rate.""")
tf.app.flags.DEFINE_integer('max_steps', 400000,
                            """Max training iteration.""")
tf.app.flags.DEFINE_integer('display', 10,
                            """Interval of loginfo display.""")
tf.app.flags.DEFINE_integer('stepvalue', 80000,
                            """Step interval to decay learning rate.""")
tf.app.flags.DEFINE_integer('snapshot', 20000,
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
tf.app.flags.DEFINE_string('finetune_folder', '', """Path to output.""")
tf.app.flags.DEFINE_string('oflownet', '', """Path to OFlowNet model.""")
tf.app.flags.DEFINE_string('scoordnet', '', """Path to SCoordNet model.""")

def get_indexes(image_num):
    def _get_group_from_range(start, end):
        group = []
        for i in range(start, end - 1):
            group.append([i, i+1])
        return group

    groups = _get_group_from_range(0, image_num)
    print 'Groups of consecutive image pairs for training:', groups
    return groups

def data_augmentation(image, coord_map, spec):
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

    concat_data = tf.concat([image, coord_map], axis=-1)  # BxHxWx9
    concat_data = image_augmentation(concat_data, spec)  # BxHxWx9

    image_crop = tf.slice(concat_data, [0, 0, 0, 0], [-1, -1, -1, 3])  # BxHxWx3
    coord_map_crop = tf.slice(concat_data, [0, 0, 0, 3], [-1, -1, -1, 3])  # BxHxWx3
    mask_map_crop = tf.slice(concat_data, [0, 0, 0, 6], [-1, -1, -1, 1])  # BxHxWx1

    mask_map_crop = tf.cast((mask_map_crop >= 1.0), tf.float32)  # BxHxWx1

    coord_map_crop = tf.concat((coord_map_crop, mask_map_crop), axis=-1) # BxHxWx4

    return image_crop, coord_map_crop

def get_training_data(image_list, label_list, spec, is_training=True):
    image_paths = read_lines(image_list)
    label_paths = read_lines(label_list)

    groups = get_indexes(len(image_paths))
    indexes = [str(i) for i in range(len(groups))]
    groups = tf.stack(groups, axis=0, name='group_indexes')

    with tf.name_scope('data_queue'):
        with tf.device('/CPU:0'):
            index_queue = tf.train.string_input_producer(indexes, shuffle=(is_training or FLAGS.shuffle))
            index = tf.string_to_number(index_queue.dequeue(), out_type=tf.int32)
            group = tf.squeeze(tf.slice(groups, [index, 0], [1, -1]), name='group') # B
            group_image_paths = tf.gather(image_paths, group)
            group_label_paths = tf.gather(label_paths, group)

            images = []
            labels = []
            for i in range(2):
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

            if is_training:
                images, labels = data_augmentation(images, labels, spec)

            coords = tf.slice(labels, [0, 0, 0, 0], [-1, -1, -1, 3])
            masks = tf.slice(labels, [0, 0, 0, 3], [-1, -1, -1, 1])

        if is_training:
            num_threads = 40
        else:
            num_threads = 1
        with tf.device('/CPU:0'):
            return tf.train.batch(
                [images, coords, masks, group],
                batch_size=spec.batch_size,
                capacity=spec.batch_size * 4,
                enqueue_many=True,
                num_threads=num_threads)

def run(image_list, label_list, spec, is_training=True):
    with tf.name_scope('data'):
        images, gt_coords, masks, group_indexes = get_training_data(image_list, label_list, spec, is_training)

    with tf.device('/device:CPU:0'):
        gt_coords = tf.image.resize_nearest_neighbor(gt_coords, [spec.image_size[0] / 8, spec.image_size[1] / 8])
        masks = tf.image.resize_nearest_neighbor(masks, [spec.image_size[0] / 8, spec.image_size[1] / 8])
        masks = tf.cast(tf.equal(masks, 1.0), tf.float32)

        if is_training and FLAGS.noise:
            mean = 0.0
            stddev = tf.squeeze(tf.random_uniform([1], 0.005, 0.01, dtype=tf.float32))
        elif FLAGS.noise:
            mean = 0.0
            stddev = tf.constant(0.02)
        else:
            mean = 0.0
            stddev = tf.constant(0.00)

        batch_size, height, width, _ = gt_coords.get_shape().as_list()
        uncertainty_shape = [batch_size, height, width, 1]
        noise = tf.random_normal(gt_coords.get_shape(), mean, stddev)
        noise_gt_coords = tf.add(gt_coords, noise, name='noisy_gt_coords')
        gt_uncertainty = tf.fill(uncertainty_shape, stddev)

        shift_masks = []
        for i in range(8):
            for j in range(8):
                offset = [-(j - 4), -(i - 4)]
                shift_mask = tf.contrib.image.translate(masks, offset, interpolation='NEAREST')
                shift_masks.append(shift_mask)
        shift_masks = tf.concat(shift_masks, axis=-1)
        shift_masks = tf.reduce_sum(shift_masks, axis=-1, keepdims=True)
        shift_mask1 = tf.slice(shift_masks, [0, 0, 0, 0], [1, -1, -1, -1])
        shift_mask2 = tf.slice(shift_masks, [1, 0, 0, 0], [1, -1, -1, -1])
        mask1 = tf.slice(masks, [0, 0, 0, 0], [1, -1, -1, -1])
        mask2 = tf.slice(masks, [1, 0, 0, 0], [1, -1, -1, -1])
        mask1 = tf.cast(((shift_mask2 + mask1) >= 65), tf.float32)
        mask2 = tf.cast(((shift_mask1 + mask2) >= 65), tf.float32)
        shift_masks = tf.concat([mask1, mask2], axis=0, name='masks')

    kfnet = KFNet(images, noise_gt_coords, gt_uncertainty, spec.focal_x, spec.focal_y, spec.u, spec.v,
                    train_scoordnet=False, train_oflownet=is_training, reuse=tf.AUTO_REUSE)

    with tf.name_scope('loss'):
        measure_coord_loss, measure_coord_accuracy = kfnet.CoordLossWithUncertainty(noise_gt_coords, gt_uncertainty, gt_coords, masks)
        temp_coord_loss, temp_coord_accuracy = kfnet.TemporalCoordLoss(gt_coords, shift_masks)

        images = tf.image.resize_bilinear(images, [spec.image_size[0] // 8, spec.image_size[1] // 8])
        temp_coord_map, temp_uncertainty_map, temp_images, _ = kfnet.GetTemporalCoord()
        image_loss = 2.0 * PhotometricLoss(images, temp_images, masks)

        smooth_loss = 50 * kfnet.SmoothLoss(temp_coord_map, images, shift_masks)

        loss = temp_coord_loss + smooth_loss + image_loss

        with tf.device('/device:CPU:0'):
            tf.summary.scalar('loss', loss)
            tf.summary.scalar('temp_loss', temp_coord_loss)
            tf.summary.scalar('temp_accuracy', temp_coord_accuracy)

    return kfnet, loss, measure_coord_loss, measure_coord_accuracy, temp_coord_loss, temp_coord_accuracy, smooth_loss, \
           image_loss, group_indexes, gt_coords * shift_masks, shift_masks, stddev

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
    variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)
    restorer = tf.train.Saver(variables)
    restorer.restore(sess, snapshot)

def train(image_list, label_list, out_dir, \
          snapshot=None, step=0, debug=False, gpu=0):

    print image_list
    image_paths = read_lines(image_list)
    spec = KFNetDataSpec()
    spec.image_num = len(image_paths)
    spec.sequence_length = 500
    spec.num_sequence = spec.image_num // spec.sequence_length
    FLAGS.stepvalue = 100000
    FLAGS.max_steps = FLAGS.stepvalue * 5
    print "----------------------------------"
    print "training image number: ", len(image_paths)
    print "batch size: ", spec.batch_size
    print "step value: ", FLAGS.stepvalue
    print "max steps: ", FLAGS.max_steps
    print "current step: ", step
    print "with noise: ", FLAGS.noise
    print "----------------------------------"

    with tf.device('/device:GPU:%d' % gpu):
        _, loss, mea_loss, mea_accuracy, temp_loss, temp_accuracy, smooth_loss, image_loss, group_indexes, _, masks, stddev, _ \
            = run(image_list, label_list, spec)
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

        summary_writer = tf.summary.FileWriter(FLAGS.model_folder+'/log', sess.graph)

        # Initialize variables.
        sess.run(init_op)
        sess.run(init_step)

        if snapshot:
            print('Pre-trained model restored from %s' % (snapshot))
            RestoreFromScope(sess, snapshot, None)

        sess.run(init_step)

        # Start populating the queue.
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        while step <= FLAGS.max_steps:
            start_time = time.time()
            summary, _, lr, out_loss, out_mea_loss, out_mea_accuracy, out_temp_loss, out_temp_accuracy, \
            out_image_loss, out_smooth_loss, out_group_indexes, out_masks, out_stddev = \
                sess.run([summary_op, optimizer, lr_op, loss, mea_loss, mea_accuracy, temp_loss, temp_accuracy, \
                          image_loss, smooth_loss, group_indexes, masks, stddev])
            duration = time.time() - start_time

            # Print info.
            if step % FLAGS.display == 0 or not FLAGS.is_training:
                summary_writer.add_summary(summary, step)
                epoch = step // (len(image_paths) * 2)
                format_str = '[%s] epoch %d, step %d, %5d~%5d, l = %.4f, l_mea = %.4f, a_mea = %.4f, l_temp = %.4f, a_temp = %.4f, ' \
                             'l_image = %.4f, l_smooth = %.4f, #pixels = %d, stddev = %.3f, lr = %.6f (%.3f sec/step)'
                print(format_str % (datetime.now(), epoch, step, out_group_indexes[0], out_group_indexes[1],
                                                 out_loss, out_mea_loss, out_mea_accuracy, out_temp_loss, out_temp_accuracy,
                                                 out_image_loss, out_smooth_loss,
                                                 np.sum(out_masks), out_stddev, lr, duration))

            # Save the model checkpoint periodically.
            if step % FLAGS.snapshot == 0 or step == FLAGS.max_steps:
                checkpoint_path = os.path.join(out_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)
            step += 1

        coord.request_stop()
        coord.join(threads)

def main(_):
    if os.path.isdir(FLAGS.finetune_folder):
        snapshot, step = get_snapshot(FLAGS.finetune_folder)
        step = 0
    else:
        snapshot, step = get_snapshot(FLAGS.model_folder)

    image_list = os.path.join(FLAGS.input_folder, 'image_list.txt')
    label_list = os.path.join(FLAGS.input_folder, 'label_list.txt')

    train(image_list, label_list, FLAGS.model_folder,
          snapshot, step, FLAGS.debug, FLAGS.gpu)


if __name__ == '__main__':
    tf.app.run()
