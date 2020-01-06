import sys
sys.path.append('/run/media/larry/fafb882a-0878-4e0a-9ccb-2fb979b7f717/e3dengine/tfmatch')
import tensorflow as tf
from tensorflow.python import debug as tf_debug
from SfMNet import *
from tools.io import read_lines, get_snapshot
from train import FLAGS, run, RestoreFromScope, get_pixel_map
from tools.common import Notify
import matplotlib.pyplot as plt
import os
import random

def eval(image_list, label_list, snapshot, out_dir):

    print image_list
    image_paths = read_lines(image_list)

    spec = SfMNetDataSpec()
    spec.scene = FLAGS.scene
    spec.batch_size = 2
    spec.image_num = len(image_paths)
    spec.sequence_length = 500
    spec.num_sequence = spec.image_num // spec.sequence_length

    print "----------------------------------"
    print "scene: ", spec.scene
    print "image number: ", len(image_paths)
    print "sequence length: ", spec.sequence_length
    print "num sequence: ", spec.num_sequence
    print "----------------------------------"

    image_paths = read_lines(image_list)
    label_paths = read_lines(label_list)

    sfmnet, loss, measure_loss, measure_accuracy, temp_loss, temp_accuracy, KF_loss, KF_accuracy, \
    group_indexes, gt_coords, masks = run(image_list, label_list, spec, False)
    images = sfmnet.GetInputImages()
    gt_coord2 = tf.slice(gt_coords, [1, 0, 0, 0], [1, -1, -1, -1])
    mask2 = tf.slice(masks, [1, 0, 0, 0], [1, -1, -1, -1])

    # pumpkin
    transform = tf.constant([[0.984219, 0.0694752, -0.162745, -0.489651],
                              [-0.0569574, 0.995137, 0.080364, -1.41071],
                              [0.167537, -0.0698263, 0.98339, 2.32241],
                              [0, 0, 0, 1]])

    config = tf.ConfigProto(log_device_placement=False)
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        if FLAGS.debug:
            sess = tf_debug.LocalCLIDebugWrapperSession(sess)

        # Start populating the queue.
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        if snapshot:
            RestoreFromScope(sess, snapshot, None)

        last_coord_map = None
        last_uncertainty_map = None
        for i in range(len(image_paths)):
            if i == 0:
                KF_coord_map, KF_uncertainty_map = sfmnet.GetMeasureCoord2()
            else:
                KF_coord_map, KF_uncertainty_map = sfmnet.GetKFCoord2(last_coord_map, last_uncertainty_map)
            last_coord_map = KF_coord_map
            last_uncertainty_map = KF_uncertainty_map

            loss, accuracy = sfmnet.CoordLossWithUncertainty(KF_coord_map, KF_uncertainty_map, gt_coord2, mask2)

            out_group_indexes, out_images, out_gt_coord2, out_mask2, \
            out_KF_coord, out_KF_uncertainty,\
            out_loss, out_accuracy \
                = sess.run([group_indexes, images, gt_coord2, mask2,
                        KF_coord_map, KF_uncertainty_map,
                        loss, accuracy])

            diff_KF_coord = np.square(out_KF_coord - out_gt_coord2)
            diff_KF_coord = np.sqrt(np.sum(diff_KF_coord, axis=-1) * out_mask2[:, :, :, 0]) * 100.  # cm
            avg_dist = np.sum(diff_KF_coord) / (np.sum(out_mask2) + 1)

            format_str = '%d~%d, loss = %.3f, accuracy = %.3f, dist = %.3f'
            print(Notify.INFO, format_str % (out_group_indexes[0], out_group_indexes[1], out_loss, out_accuracy, avg_dist))

            if FLAGS.show:
                fig, axarr = plt.subplots(3, 2)
                plt.subplot(3, 2, 1)
                plt.imshow(out_images[0, :, :, :] / 255.0)
                plt.subplot(3, 2, 2)
                plt.imshow(out_images[1, :, :, :] / 255.0)
                plt.subplot(3, 2, 3)
                plt.imshow(out_KF_coord[0, :, :, :])
                plt.subplot(3, 2, 4)
                plt.imshow(out_gt_coord2[0, :, :, :])
                plt.subplot(3, 2, 5)
                plt.imshow(diff_KF_coord[0, :, :])
                plt.subplot(3, 2, 6)
                plt.imshow(1.0 / out_KF_uncertainty[0, :, :, 0])
                plt.show()


        coord.request_stop()
        coord.join(threads)

def main(_):
    snapshot, step = get_snapshot(FLAGS.model_folder)

    image_list = os.path.join(FLAGS.input_folder, 'image_list.txt')
    label_list = os.path.join(FLAGS.input_folder, 'label_list.txt')

    eval(image_list, label_list, snapshot, FLAGS.output_folder)


if __name__ == '__main__':
    tf.app.run()