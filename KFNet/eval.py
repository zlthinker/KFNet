import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from tensorflow.python import debug as tf_debug
from KFNet import *
from tools.io import read_lines, get_snapshot
from train import FLAGS, run, RestoreFromScope
import matplotlib.pyplot as plt

def dist_error(coords, gt_coords, mask):
    """
    :param coords: HxWx3
    :param gt_coords: HxWx3
    :param mask: HxWx1
    :return: HxW
    """
    dists = np.square(coords - gt_coords)
    dists = np.sqrt(np.sum(dists, axis=-1))
    dists = dists * mask[:, :, 0]
    data_array = dists.flatten().tolist()
    data_array = filter(lambda x: x > 0, data_array)
    return np.mean(data_array) * 100.0, dists * 100

def eval(image_list, label_list, pose_file, snapshot, out_dir):

    print image_list
    image_paths = read_lines(image_list)

    spec = KFNetDataSpec()
    spec.scene = FLAGS.scene
    spec.batch_size = 4
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

    kfnet, loss, measure_loss, measure_accuracy, temp_loss, temp_accuracy, KF_loss, KF_accuracy, \
    group_indexes, gt_coords, masks = run(image_list, label_list, pose_file, spec, False)

    images = kfnet.GetInputImages()
    measure_coord_map, measure_uncertainty_map = kfnet.GetMeasureCoord()
    measure_uncertainty_map = 1.0 / measure_uncertainty_map
    temp_coord_map, temp_uncertainty_map, KF_coord_map, KF_uncertainty_map = kfnet.GetKFCoord()
    temp_uncertainty_map = 1.0 / temp_uncertainty_map
    KF_uncertainty_map = 1.0 / KF_uncertainty_map
    NIS = kfnet.GetNIS()

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
    measure_coord_map = ApplyTransform(measure_coord_map, transform)
    temp_coord_map = ApplyTransform(temp_coord_map, transform)
    KF_coord_map = ApplyTransform(KF_coord_map, transform)
    gt_coords = ApplyTransform(gt_coords, transform)

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

        if os.path.isdir(FLAGS.output_folder):
            measure_folder = os.path.join(FLAGS.output_folder, 'measure_match')
            if not os.path.isdir(measure_folder):
                os.mkdir(measure_folder)
            temp_folder = os.path.join(FLAGS.output_folder, 'temp_match')
            if not os.path.isdir(temp_folder):
                os.mkdir(temp_folder)
            KF_folder = os.path.join(FLAGS.output_folder, 'KF_match')
            if not os.path.isdir(KF_folder):
                os.mkdir(KF_folder)

        dists_m = []
        dists_t = []
        dists_kf = []
        errors = []
        percents = []
        for i in range(len(image_paths)):
            out_group_indexes, out_images, out_gt_coord, out_masks, \
            out_measure_coord, out_measure_uncertainty, \
            out_temp_coord, out_temp_uncertainty, \
            out_KF_coord, out_KF_uncertainty, \
            out_measure_loss, out_measure_accuracy, \
            out_temp_loss, out_temp_accuracy, \
            out_KF_loss, out_KF_accuracy,\
            out_NIS = sess.run([group_indexes, images, gt_coords, masks,
                                                     measure_coord_map, measure_uncertainty_map,
                                                     temp_coord_map, temp_uncertainty_map,
                                                     KF_coord_map, KF_uncertainty_map,
                                                     measure_loss, measure_accuracy,
                                                     temp_loss, temp_accuracy,
                                                     KF_loss, KF_accuracy,
                                                     NIS])

            out_NIS = out_NIS * np.tile(out_masks, [1, 1, 1, 3])
            out_NIS = out_NIS[:, ::4, ::4, :]
            NIS_array = out_NIS.flatten().tolist()
            NIS_array = filter(lambda x: x > 0.0, NIS_array)
            valid_NIS_array = filter(lambda x: x > 0.0157 and x < 2.706, NIS_array)
            percent = len(valid_NIS_array) / float(len(NIS_array))
            # print 'Valid NIS percent', percent
            # sns.distplot(NIS_array, fit=stats.chi2)

            median_mea_dist, diff_coords = dist_error(out_measure_coord[-1, :, :, :], out_gt_coord[-1, :, :, :],
                                                      out_masks[-1, :, :, :])
            median_temp_dist, diff_temp_coords = dist_error(out_temp_coord[-1, :, :, :], out_gt_coord[-1, :, :, :],
                                                            out_masks[-1, :, :, :])
            median_KF_dist, diff_KF_coords = dist_error(out_KF_coord[-1, :, :, :], out_gt_coord[-1, :, :, :],
                                                        out_masks[-1, :, :, :])

            dists_m.append(median_mea_dist)
            dists_t.append(median_temp_dist)
            dists_kf.append(median_KF_dist)

            percents.append(percent)
            errors.append(out_measure_loss)

            format_str = 'step %d, %5d~%5d~%5d~%5d, l_mea=%.3f, l_temp=%.3f, l_KF= %.3f, ' \
                         'a_mea=%.3f, a_temp=%.3f,a_KF=%.3f, d_mea=%.3f, d_temp=%.3f, d_KF=%.3f, per=%.3f'
            print (format_str % (i, out_group_indexes[0], out_group_indexes[1], out_group_indexes[2], out_group_indexes[3],\
                                 out_measure_loss, out_temp_loss, out_KF_loss, \
                                 out_measure_accuracy, out_temp_accuracy, out_KF_accuracy, \
                                 median_mea_dist, median_temp_dist, median_KF_dist, percent))

            if not FLAGS.show:
                if os.path.isdir(FLAGS.output_folder):
                    reg_values = np.concatenate([out_measure_coord[-1, :, :, :], out_measure_uncertainty[-1, :, :, :]], axis=-1)
                    reg_values = reg_values.astype(np.float32)
                    coord_save_path = os.path.join(measure_folder, 'coord_' + str(i) + '.npy')
                    np.save(coord_save_path, reg_values)

                    reg_values = np.concatenate([out_temp_coord[-1, :, :, :], out_temp_uncertainty[-1, :, :, :]], axis=-1)
                    reg_values = reg_values.astype(np.float32)
                    coord_save_path = os.path.join(temp_folder, 'coord_' + str(i) + '.npy')
                    np.save(coord_save_path, reg_values)

                    reg_values = np.concatenate([out_KF_coord[-1, :, :, :], out_KF_uncertainty[-1, :, :, :]], axis=-1)
                    reg_values = reg_values.astype(np.float32)
                    coord_save_path = os.path.join(KF_folder, 'coord_' + str(i) + '.npy')
                    np.save(coord_save_path, reg_values)

            else:
                weight_K = out_temp_uncertainty[-1, :, :, 0] / \
                           (out_temp_uncertainty[-1, :, :, 0] + out_measure_uncertainty[-1, :, :, 0])

                fig, axarr = plt.subplots(4, 3)
                plt.subplot(4, 3, 1)
                plt.imshow(out_images[-1, :, :, :] / 255.0)
                plt.subplot(4, 3, 2)
                plt.imshow(out_gt_coord[-1, :, :, :])
                plt.subplot(4, 3, 3)
                plt.imshow(weight_K)
                plt.subplot(4, 3, 4)
                plt.imshow(np.minimum(out_measure_uncertainty[-1, :, :, 0], 1000.0))
                plt.subplot(4, 3, 5)
                plt.imshow(out_measure_coord[-1, :, :, :])
                plt.subplot(4, 3, 6)
                plt.imshow(diff_coords[:, :])
                plt.subplot(4, 3, 7)
                plt.imshow(np.minimum(out_temp_uncertainty[-1, :, :, 0], 1000.0))
                plt.subplot(4, 3, 8)
                plt.imshow(out_temp_coord[-1, :, :, :])
                plt.subplot(4, 3, 9)
                plt.imshow(diff_temp_coords[:, :])
                plt.subplot(4, 3, 10)
                plt.imshow(np.minimum(out_KF_uncertainty[-1, :, :, 0], 1000.0))
                plt.subplot(4, 3, 11)
                plt.imshow(out_KF_coord[-1, :, :, :])
                plt.subplot(4, 3, 12)
                plt.imshow(diff_KF_coords[:, :])

                # save_path = os.path.join(FLAGS.output_folder, str(i)+'.png')
                # plt.savefig(save_path)
                plt.show()
                plt.close(fig)

        plt.scatter(errors, percents)
        plt.show()

        coord.request_stop()
        coord.join(threads)

        print 'Mean dist (measure/temp/KF):', np.mean(dists_m), np.mean(dists_t), np.mean(dists_kf)

def main(_):
    snapshot, step = get_snapshot(FLAGS.model_folder)

    image_list = os.path.join(FLAGS.input_folder, 'image_list.txt')
    label_list = os.path.join(FLAGS.input_folder, 'label_list.txt')
    pose_file = os.path.join(FLAGS.input_folder, 'poses.bin')

    eval(image_list, label_list, pose_file, snapshot, FLAGS.output_folder)


if __name__ == '__main__':
    tf.app.run()