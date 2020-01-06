import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from train import *
import matplotlib.pyplot as plt

def SetVariableByName(sess, name, val):
    var = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)[0]
    sess.run(var.assign(val))

def get_NIS_measurement(out_NIS):
    NIS_array = out_NIS.flatten().tolist()
    NIS_array = filter(lambda x: x > 0.0, NIS_array)
    valid_NIS_array = filter(lambda x: x > 0.0157 and x < 2.706, NIS_array)
    percent = len(valid_NIS_array) / float(len(NIS_array))
    return percent

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
    return np.median(data_array) * 100.0, dists * 100

def eval(image_list, label_list, snapshot):

    print image_list
    print label_list
    image_paths = read_lines(image_list)
    label_paths = read_lines(label_list)
    assert(len(image_paths) == len(label_paths))

    spec = KFNetDataSpec()
    spec.scene = FLAGS.scene
    spec.batch_size = 2
    spec.image_num = len(image_paths)

    print "----------------------------------"
    print "scene: ", spec.scene
    print "image number: ", len(image_paths)
    print "----------------------------------"

    last_coord = tf.get_variable('last_coord', [1, spec.image_size[0]//8, spec.image_size[1]//8, 3], trainable=False)
    last_uncertainty = tf.get_variable('last_uncertainty', [1, spec.image_size[0]//8, spec.image_size[1]//8, 1], trainable=False)
    measure_coord, measure_uncertainty, temp_coord, temp_uncertainty, KF_coord, KF_uncertainty, NIS, indexes, \
    measure_loss, measure_accuracy, temp_loss, temp_accuracy, KF_loss, KF_accuracy, gt_coords, masks \
        = KF_fusion(image_list, label_list, last_coord, last_uncertainty, spec)
    transform = get_7scene_transform()
    trans_measure_coord = ApplyTransform(measure_coord, transform)
    trans_temp_coord = ApplyTransform(temp_coord, transform)
    trans_KF_coord = ApplyTransform(KF_coord, transform)
    init_op = tf.global_variables_initializer()
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

    config = tf.ConfigProto(log_device_placement=False)
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        sess.run(init_op)
        if snapshot:
            RestoreFromScope(sess, snapshot, 'ScoreNet')
            RestoreFromScope(sess, snapshot, 'Temporal')

        # Start populating the queue.
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        dists_m = []
        dists_t = []
        dists_kf = []
        for i in range(len(image_paths)):
            out_indexes, out_measure_loss, out_measure_accuracy, out_temp_loss, out_temp_accuracy, out_KF_loss, out_KF_accuracy, out_NIS, \
            out_measure_coord, out_measure_uncertainty, out_temp_coord, out_temp_uncertainty, out_KF_coord, out_KF_uncertainty, \
            out_trans_measure_coord, out_trans_temp_coord, out_trans_KF_coord, out_gt_coords, out_masks = \
            sess.run([indexes, measure_loss, measure_accuracy, temp_loss, temp_accuracy, KF_loss, KF_accuracy, NIS,
                      measure_coord, measure_uncertainty, temp_coord, temp_uncertainty, KF_coord, KF_uncertainty,
                      trans_measure_coord, trans_temp_coord, trans_KF_coord, gt_coords, masks])

            out_NIS_measure = get_NIS_measurement(out_NIS)

            if FLAGS.NIS:
                out_NIS = np.sum(out_NIS, axis=-1)
                NIS_mask = (out_NIS > 7.815)
                NIS_mask = NIS_mask.astype(float)
                NIS_mask = np.stack([NIS_mask, NIS_mask, NIS_mask], axis=-1)
                out_trans_KF_coord = NIS_mask * out_trans_measure_coord + (1.0 - NIS_mask) * out_trans_KF_coord

            if i % spec.sequence_length == 0:
                print 'Reinitialize at the begin of a new sequence.'
                SetVariableByName(sess, 'last_coord', out_measure_coord)
                SetVariableByName(sess, 'last_uncertainty', out_measure_uncertainty)
                out_trans_temp_coord = out_trans_measure_coord
                out_temp_uncertainty = out_measure_uncertainty
                out_trans_KF_coord = out_trans_measure_coord
                out_KF_uncertainty = out_measure_uncertainty
            else:
                SetVariableByName(sess, 'last_coord', out_KF_coord)
                SetVariableByName(sess, 'last_uncertainty', out_KF_uncertainty)

            dist_m, dist_map_m = dist_error(out_trans_measure_coord[-1, :, :, :], out_gt_coords[-1, :, :, :], out_masks[-1, :, :, :])
            dist_t, dist_map_t = dist_error(out_trans_temp_coord[-1, :, :, :], out_gt_coords[-1, :, :, :], out_masks[-1, :, :, :])
            dist_kf, dist_map_kf = dist_error(out_trans_KF_coord[-1, :, :, :], out_gt_coords[-1, :, :, :], out_masks[-1, :, :, :])
            dists_m.append(dist_m)
            dists_t.append(dist_t)
            dists_kf.append(dist_kf)

            format_str = "%d, frame %d~%d, l_m = %.3f, l_t = %.3f, l_kf = %.3f, a_m = %.3f, a_t = %.3f, a_kf = %.3f, " \
                         "d_m = %.3f, d_t = %.3f, d_kf = %.3f, nis = %.3f"
            index = out_indexes[1]
            print (format_str % (i, out_indexes[0], out_indexes[1], out_measure_loss, out_temp_loss, out_KF_loss,
                                 out_measure_accuracy, out_temp_accuracy, out_KF_accuracy, dist_m, dist_t, dist_kf,
                                 out_NIS_measure))
            # continue

            if not FLAGS.show:
                if os.path.isdir(FLAGS.output_folder):
                    reg_values = np.concatenate([out_trans_measure_coord[-1, :, :, :], 1.0/out_measure_uncertainty[-1, :, :, :]], axis=-1)
                    reg_values = reg_values.astype(np.float32)
                    coord_save_path = os.path.join(measure_folder, 'coord_' + str(index) + '.npy')
                    np.save(coord_save_path, reg_values)

                    reg_values = np.concatenate([out_trans_temp_coord[-1, :, :, :], 1.0/out_temp_uncertainty[-1, :, :, :]], axis=-1)
                    reg_values = reg_values.astype(np.float32)
                    coord_save_path = os.path.join(temp_folder, 'coord_' + str(index) + '.npy')
                    np.save(coord_save_path, reg_values)

                    reg_values = np.concatenate([out_trans_KF_coord[-1, :, :, :], 1.0/out_KF_uncertainty[-1, :, :, :]], axis=-1)
                    reg_values = reg_values.astype(np.float32)
                    coord_save_path = os.path.join(KF_folder, 'coord_' + str(index) + '.npy')
                    np.save(coord_save_path, reg_values)
            else:
                weight_K = out_temp_uncertainty[-1, :, :, 0] / \
                           (out_temp_uncertainty[-1, :, :, 0] + out_measure_uncertainty[-1, :, :, 0])

                fig, axarr = plt.subplots(4, 3)
                plt.subplot(4, 3, 1)
                plt.imshow(out_masks[-1, :, :, 0])
                plt.subplot(4, 3, 2)
                plt.imshow(out_gt_coords[-1, :, :, :])
                plt.subplot(4, 3, 3)
                plt.imshow(np.minimum(out_NIS[-1, :, :], 20))
                plt.subplot(4, 3, 4)
                plt.imshow(np.minimum(out_measure_uncertainty[-1, :, :, 0], 255.0))
                plt.subplot(4, 3, 5)
                plt.imshow(out_trans_measure_coord[-1, :, :, :])
                plt.subplot(4, 3, 6)
                plt.imshow(np.minimum(dist_map_m[:, :], 10))
                plt.subplot(4, 3, 7)
                plt.imshow(np.minimum(out_temp_uncertainty[-1, :, :, 0], 255.0))
                plt.subplot(4, 3, 8)
                plt.imshow(out_trans_temp_coord[-1, :, :, :])
                plt.subplot(4, 3, 9)
                plt.imshow(np.minimum(dist_map_t[:, :], 10))
                plt.subplot(4, 3, 10)
                plt.imshow(np.minimum(out_KF_uncertainty[-1, :, :, 0], 255.0))
                plt.subplot(4, 3, 11)
                plt.imshow(out_trans_KF_coord[-1, :, :, :])
                plt.subplot(4, 3, 12)
                plt.imshow(np.minimum(dist_map_kf[:, :], 10))
                plt.show()
                plt.close(fig)

        coord.request_stop()
        coord.join(threads)

        print 'Median dist error: ', np.median(dists_m), np.median(dists_t), np.median(dists_kf)
        print 'Mean dist error: ', np.mean(dists_m), np.mean(dists_t), np.mean(dists_kf)
        print 'stddev error: ', np.std(dists_m), np.std(dists_t), np.std(dists_kf)


def main(_):
    snapshot, step = get_snapshot(FLAGS.model_folder)

    image_list = os.path.join(FLAGS.input_folder, 'image_list.txt')
    label_list = os.path.join(FLAGS.input_folder, 'label_list.txt')

    eval(image_list, label_list, snapshot)

if __name__ == '__main__':
    tf.app.run()