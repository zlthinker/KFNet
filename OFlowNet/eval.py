import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from tensorflow.python import debug as tf_debug
from KFNet import *
from tools.io import read_lines, get_snapshot
from train import FLAGS, run, RestoreFromScope
import matplotlib.pyplot as plt

def eval(image_list, label_list, snapshot, out_dir):

    print image_list
    image_paths = read_lines(image_list)

    spec = KFNetDataSpec()
    spec.scene = 'stairs'
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

    kfnet, loss, _, _, temp_loss, temp_accuracy, smooth_loss, image_loss, group_indexes, gt_coords, masks, stddev, temp_images \
        = run(image_list, label_list, spec, False)

    images = kfnet.GetInputImages()
    temp_coord_map, temp_uncertainty_map, temp_prob = kfnet.GetTemporalCoord()
    temp_uncertainty_map = 1.0 / temp_uncertainty_map

    config = tf.ConfigProto(log_device_placement=False)
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        if FLAGS.debug:
            sess = tf_debug.LocalCLIDebugWrapperSession(sess)

        # Start populating the queue.
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        if snapshot:
            print('Pre-trained model restored from %s' % (snapshot))
            RestoreFromScope(sess, snapshot, None)

        losses = []
        accuracies = []
        diff_coords = []
        for i in range(len(image_paths)):
            out_images, out_gt_coord, out_masks, out_group_indexes, \
            out_temp_coord, out_temp_uncertainty, \
            out_temp_loss, out_temp_accuracy, out_temp_prob, out_temp_images = sess.run([images, gt_coords, masks, group_indexes,
                                                        temp_coord_map, temp_uncertainty_map,
                                                        temp_loss, temp_accuracy, temp_prob, temp_images])

            diff_coord = np.square(out_temp_coord - out_gt_coord)
            diff_coord = np.sum(diff_coord, axis=-1)
            diff_coord = np.sqrt(diff_coord) * 100.0 * out_masks[:, :, :, 0]
            avg_diff_coord = np.sum(diff_coord) / np.sum(out_masks)

            losses.append(out_temp_loss)
            accuracies.append(out_temp_accuracy)
            diff_coords.append(avg_diff_coord)

            format_str = 'step %d, %5d~%5d, l_temp = %.4f, a_temp = %.4f, diff_coord = %.4f'
            print(format_str % (i, out_group_indexes[0], out_group_indexes[1], out_temp_loss, out_temp_accuracy, avg_diff_coord))
            # continue

            fig, axarr = plt.subplots(5, 2)
            plt.subplot(5, 2, 1)
            plt.imshow(out_images[0, :, :, :] / 255.0)
            plt.subplot(5, 2, 2)
            plt.imshow(out_images[1, :, :, :] / 255.0)
            plt.subplot(5, 2, 3)
            plt.imshow(out_temp_images[0, :, :, :] / 255.0)
            plt.subplot(5, 2, 4)
            plt.imshow(out_temp_images[1, :, :, :] / 255.0)
            # plt.subplot(5, 2, 3)
            # plt.imshow(out_gt_coord[0, :, :, :])
            # plt.subplot(5, 2, 4)
            # plt.imshow(out_gt_coord[1, :, :, :])
            plt.subplot(5, 2, 5)
            plt.imshow(out_temp_coord[0, :, :, :])
            plt.subplot(5, 2, 6)
            plt.imshow(out_temp_coord[1, :, :, :])
            plt.subplot(5, 2, 7)
            plt.imshow(out_temp_uncertainty[0, :, :, 0])
            plt.subplot(5, 2, 8)
            plt.imshow(out_temp_uncertainty[1, :, :, 0])
            plt.subplot(5, 2, 9)
            plt.imshow(diff_coord[0, :, :])
            plt.subplot(5, 2, 10)
            plt.imshow(diff_coord[1, :, :])

            plt.show()

        print 'Median loss', np.median(losses)
        print 'Median accuracy', np.median(accuracies)
        print 'Median diff coord', np.median(diff_coords)

        coord.request_stop()
        coord.join(threads)

def main(_):
    snapshot, step = get_snapshot(FLAGS.model_folder)

    image_list = os.path.join(FLAGS.input_folder, 'image_list.txt')
    label_list = os.path.join(FLAGS.input_folder, 'label_list.txt')

    eval(image_list, label_list, snapshot, FLAGS.output_folder)


if __name__ == '__main__':
    tf.app.run()