import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from model import run_testing, get_indexes, read_lines, FLAGS
from tools.io import get_snapshot
from tensorflow.python import debug as tf_debug
from cnn_wrapper import helper, SCoordNet
import matplotlib.pyplot as plt
from util import *

def dist_error(coords, gt_coords, mask):
    """
    :param coords: HxWx3
    :param gt_coords: HxWx3
    :param mask: HxWx1
    :return: HxW
    """
    dists = np.square(coords - gt_coords)
    dists = np.sqrt(np.sum(dists, axis=-1))
    dists = dists * mask[:, :, 0] * 100 # cm
    data_array = dists.flatten().tolist()
    data_array = filter(lambda x: x > 0, data_array)
    accurate_data_array = filter(lambda x: x < 2, data_array)
    accuracy = len(accurate_data_array) / float(len(data_array))
    return np.median(data_array), dists, accuracy

def median_uncertainty(uncertainty, mask):
    uncertainty = uncertainty * mask
    data_array = uncertainty.flatten().tolist()
    data_array = filter(lambda x: x > 0, data_array)
    return np.median(data_array) * 100.0

def eval(image_list, label_list, transform_file, output_folder, snapshot, debug=False):
    print image_list
    print label_list

    image_paths = read_lines(image_list)
    image_num = len(image_paths)

    spec = helper.get_data_spec(model_class=SCoordNet)
    spec.batch_size = 1
    spec.scene = FLAGS.scene
    if FLAGS.scene == 'GreatCourt' or FLAGS.scene == 'KingsCollege' or \
        FLAGS.scene == 'OldHospital' or FLAGS.scene == 'ShopFacade' or \
        FLAGS.scene == 'StMarysChurch' or FLAGS.scene == 'Street' or \
        FLAGS.scene == 'Street-east' or FLAGS.scene == 'Street-west' or \
        FLAGS.scene == 'Street-south' or FLAGS.scene == 'Street-north1' or \
        FLAGS.scene == 'Street-north2':
        spec.image_size = (480, 848)
        spec.crop_size = (480, 848)
    elif FLAGS.scene == 'DeepLoc':
        spec.image_size = (480, 864)
        spec.crop_size = (480, 864)

    indexes = get_indexes(image_num)
    print indexes
    scoordnet, images, coords, uncertainty, gt_coords, mask, image_indexes = \
        run_testing(indexes, image_list, label_list, transform_file, spec)

    # configuration
    config = tf.ConfigProto(log_device_placement=False)
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        # get_num_flops(sess)

        if debug:
            sess = tf_debug.LocalCLIDebugWrapperSession(sess)

        # Start populating the queue.
        coordinator = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coordinator)

        print('Pre-trained model restored from %s' % (snapshot))
        restore_variables = tf.global_variables()
        # restore_variables = [var for var in restore_variables if var.name[0:6] == 'weight']
        restorer = tf.train.Saver(restore_variables)
        restorer.restore(sess, snapshot)

        loop_num = len(indexes) / spec.batch_size
        coeffs = []
        dists = []
        median_uncertainties = []
        accuracies = []

        for i in range(loop_num):
            batch_images, batch_coords, batch_uncertainty, batch_gt_coords, batch_mask, batch_image_indexes = \
                sess.run([images, coords, uncertainty, gt_coords, mask, image_indexes])

            for b in range(spec.batch_size):
                id = i * spec.batch_size + b
                index = indexes[id]

                out_images = batch_images[b, :, :, :]
                out_uncertainty = batch_uncertainty[b, :, :, :]
                out_mask = batch_mask[b, :, :, :]
                out_coords = batch_coords[b, :, :, :]
                out_gt_coords = batch_gt_coords[b, :, :, :]
                out_weights = 1.0 / out_uncertainty
                out_image_index = batch_image_indexes[b]

                median_dist, euc_diff_coords, accuracy = dist_error(out_coords, out_gt_coords, out_mask)
                dists.append(median_dist)
                med_uncertainty = median_uncertainty(out_uncertainty, out_mask)
                median_uncertainties.append(med_uncertainty)
                accuracies.append(accuracy)

                print index, out_image_index, ', median dist:', median_dist, \
                    ', median uncertainty:', med_uncertainty, \
                    ', accuracy:', accuracy

                coord_save_path = os.path.join(output_folder, 'coord_' + str(index) + '.npy')
                reg_values = np.concatenate([out_coords, out_weights], axis=-1)
                reg_values = reg_values.astype(np.float32)
                np.save(coord_save_path, reg_values)

                if FLAGS.show:
                    fig1, axarr = plt.subplots(2, 3)
                    plt.subplot(2, 3, 1)
                    plt.imshow(out_images / 255.0)
                    plt.subplot(2, 3, 2)
                    plt.imshow(out_coords)
                    plt.subplot(2, 3, 3)
                    plt.imshow(out_gt_coords)
                    plt.subplot(2, 3, 4)
                    uncertainty_map = plt.imshow(out_weights[:, :, 0])
                    fig1.colorbar(uncertainty_map)
                    plt.subplot(2, 3, 5)
                    dist_map = plt.imshow(euc_diff_coords)
                    fig1.colorbar(dist_map)

                    plt.show()
                    plt.close(fig1)

        print 'Median dist error', np.median(dists)
        print 'Mean dist error', np.mean(dists)
        print 'stddev error', np.std(dists)
        print 'Median uncertainty', np.median(median_uncertainties)
        print 'Mean accuracy', np.mean(accuracies)
        print '--------------------------------------------------'

        coordinator.request_stop()
        coordinator.join(threads)

def main(_):
    snapshot, step = get_snapshot(FLAGS.model_folder)

    image_list = os.path.join(FLAGS.input_folder, 'image_list.txt')
    label_list = os.path.join(FLAGS.input_folder, 'label_list.txt')
    transform_file = os.path.join(FLAGS.input_folder, 'transform.txt')

    eval(image_list, label_list, transform_file, FLAGS.output_folder, snapshot, FLAGS.debug)

if __name__ == '__main__':
    tf.app.run()