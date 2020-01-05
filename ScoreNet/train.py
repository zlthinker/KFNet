import sys, os, time
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import tensorflow as tf
from model import run_training, FLAGS
from tools.io import get_snapshot, get_num_trainable_params
from tensorflow.python import debug as tf_debug
from cnn_wrapper import helper, ScoreNet
from datetime import datetime

def set_stepvalue():
    if FLAGS.scene == 'chess':
        FLAGS.stepvalue = 100000
        FLAGS.max_steps = FLAGS.stepvalue * 5
    elif FLAGS.scene == 'fire':
        FLAGS.stepvalue = 30000
        FLAGS.max_steps = FLAGS.stepvalue * 5
    elif FLAGS.scene == 'heads':
        FLAGS.stepvalue = 60000
        FLAGS.max_steps = FLAGS.stepvalue * 5
    elif FLAGS.scene == 'office':
        FLAGS.stepvalue = 100000
        FLAGS.max_steps = FLAGS.stepvalue * 5
    elif FLAGS.scene == 'pumpkin':
        FLAGS.stepvalue = 100000
        FLAGS.max_steps = FLAGS.stepvalue * 5
    elif FLAGS.scene == 'redkitchen':
        FLAGS.stepvalue = 100000
        FLAGS.max_steps = FLAGS.stepvalue * 5
    elif FLAGS.scene == 'stairs':
        FLAGS.stepvalue = 100000
        FLAGS.max_steps = FLAGS.stepvalue * 5
    elif FLAGS.scene == 'apt1-kitchen':
        FLAGS.stepvalue = 30000
        FLAGS.max_steps = FLAGS.stepvalue * 5
    elif FLAGS.scene == 'apt1-living':
        FLAGS.stepvalue = 30000
        FLAGS.max_steps = FLAGS.stepvalue * 5
    elif FLAGS.scene == 'apt2-bed':
        FLAGS.stepvalue = 30000
    elif FLAGS.scene == 'apt2-kitchen':
        FLAGS.stepvalue = 30000
        FLAGS.max_steps = FLAGS.stepvalue * 5
    elif FLAGS.scene == 'apt2-living':
        FLAGS.stepvalue = 70000
        FLAGS.max_steps = FLAGS.stepvalue * 5
    elif FLAGS.scene == 'apt2-luke':
        FLAGS.stepvalue = 140000
    elif FLAGS.scene == 'office1-gates362':
        FLAGS.stepvalue = 70000
        FLAGS.max_steps = FLAGS.stepvalue * 5
    elif FLAGS.scene == 'office1-gates381':
        FLAGS.stepvalue = 140000
        FLAGS.max_steps = FLAGS.stepvalue * 5
    elif FLAGS.scene == 'office1-lounge':
        FLAGS.stepvalue = 70000
        FLAGS.max_steps = FLAGS.stepvalue * 5
    elif FLAGS.scene == 'office1-manolis':
        FLAGS.stepvalue = 70000
    elif FLAGS.scene == 'office2-5a':
        FLAGS.stepvalue = 70000
        FLAGS.max_steps = FLAGS.stepvalue * 5
    elif FLAGS.scene == 'office2-5b':
        FLAGS.stepvalue = 140000
        FLAGS.max_steps = FLAGS.stepvalue * 5
    elif FLAGS.scene == 'GreatCourt':
        FLAGS.stepvalue = 1200000
        FLAGS.max_steps = FLAGS.stepvalue * 3
    elif FLAGS.scene == 'KingsCollege':
        FLAGS.stepvalue = 200000
        FLAGS.max_steps = FLAGS.stepvalue * 3
    elif FLAGS.scene == 'OldHospital':
        FLAGS.stepvalue = 200000
        FLAGS.max_steps = FLAGS.stepvalue * 3
    elif FLAGS.scene == 'ShopFacade':
        FLAGS.stepvalue = 200000
        FLAGS.max_steps = FLAGS.stepvalue * 3
    elif FLAGS.scene == 'StMarysChurch':
        FLAGS.stepvalue = 200000
        FLAGS.max_steps = FLAGS.stepvalue * 3
    elif FLAGS.scene == 'Street':
        FLAGS.stepvalue = 3000000
        FLAGS.max_steps = FLAGS.stepvalue * 3
    elif FLAGS.scene == 'Street-east' or FLAGS.scene == 'Street-west' or FLAGS.scene == 'Street-south' \
            or FLAGS.scene == 'Street-north1' or FLAGS.scene == 'Street-north2':
        FLAGS.stepvalue = 200000
        FLAGS.max_steps = FLAGS.stepvalue * 3
    elif FLAGS.scene == 'DeepLoc':
        FLAGS.stepvalue = 200000
        FLAGS.max_steps = FLAGS.stepvalue * 3
    else:
        print 'Invalid scene:', FLAGS.scene
        exit()



def solver(loss):
    """Solver."""
    # Get weight variable list.
    weights_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    # Apply regularization to variables.
    reg_loss = tf.contrib.layers.apply_regularization(
        tf.contrib.layers.l2_regularizer(scale=FLAGS.weight_decay), weights_list)
    with tf.device('/device:CPU:0'):
        # Get global step counter.
        global_step = tf.Variable(initial_value=0, trainable=False, name='global_step')
    # Decay the learning rate exponentially based on the number of steps.
    lr_op = tf.train.exponential_decay(FLAGS.base_lr,
                                       global_step=global_step,
                                       decay_steps=FLAGS.stepvalue,
                                       decay_rate=FLAGS.gamma,
                                       name='lr')
    # Get the optimizer. Moving statistics are added to optimizer.
    bn_list = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    # For networks with batch normalization layers, it is necessary to
    # explicitly fetch its moving statistics and add them to the optimizer.
    with tf.control_dependencies(bn_list):
        opt = tf.train.AdamOptimizer(learning_rate=lr_op).minimize(
            loss + reg_loss, global_step=global_step)
    return opt, lr_op, reg_loss

def train(image_list, label_list, out_dir, \
          snapshot=None, init_step=0, debug=False):

    print image_list
    print label_list
    if FLAGS.reset_step >=0:
        init_step = FLAGS.reset_step

    spec = helper.get_data_spec(model_class=ScoreNet)
    spec.scene = FLAGS.scene
    set_stepvalue()

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

    print "--------------------------------"
    print "scene:", spec.scene
    print "batch size: ", spec.batch_size
    print "step value: ", FLAGS.stepvalue
    print "max steps: ", FLAGS.max_steps
    print "current step: ", init_step

    raw_input("Please check the meta info, press any key to continue...")

    loss, coord_loss, smooth_loss, accuracy, batch_indexes = run_training(image_list, label_list)

    print '# trainable parameters: ', get_num_trainable_params()

    with tf.device('/device:GPU:%d' % FLAGS.gpu):
        optimizer, lr_op, reg_loss = solver(loss)
        init_op = tf.global_variables_initializer()

    # configuration
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=None)
    summary_op = tf.summary.merge_all()

    config = tf.ConfigProto(log_device_placement=False)
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        if debug:
            sess = tf_debug.LocalCLIDebugWrapperSession(sess)

        summary_writer = tf.summary.FileWriter(FLAGS.model_folder + '/log', sess.graph)

        # Initialize variables.
        print('Running initializaztion operator.')
        sess.run(tf.global_variables_initializer())
        step = init_step

        # Start populating the queue.
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        if snapshot:
            print('Pre-trained model restored from %s' % (snapshot))
            restore_variables = tf.global_variables()
            restorer = tf.train.Saver(restore_variables)
            restorer.restore(sess, snapshot)
            vars = {v.name: v for v in restore_variables}
            assign_step = vars['global_step:0'].assign(tf.constant(init_step))
            sess.run(assign_step)

        while step <= FLAGS.max_steps:
            start_time = time.time()
            summary, _, out_loss, out_coord_loss, out_smooth_loss, out_accuracy, out_indexes, lr = \
                sess.run([summary_op, optimizer, loss, coord_loss, smooth_loss, accuracy, batch_indexes, lr_op])
            duration = time.time() - start_time

            # Print info.
            if step % FLAGS.display == 0 or not FLAGS.is_training:
                summary_writer.add_summary(summary, step)
                format_str = '[%s] step %d/%d, %4d~%4d~%4d~%4d, loss = %.4f, coord_loss = %.4f, smooth_loss = %.4f, accuracy = %.4f, lr = %.6f (%.3f sec/step)'
                print(format_str % (datetime.now(), step, FLAGS.max_steps,
                                                 out_indexes[-4], out_indexes[-3], out_indexes[-2], out_indexes[-1],
                                                 out_loss, out_coord_loss, out_smooth_loss, out_accuracy,
                                                 lr, duration))

            # Save the model checkpoint periodically.
            if step % FLAGS.snapshot == 0 or step == FLAGS.max_steps:
                checkpoint_path = os.path.join(out_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)
            step += 1

        coord.request_stop()
        coord.join(threads)

def main(_):
    if FLAGS.finetune_folder != '':
        snapshot, step = get_snapshot(FLAGS.finetune_folder)
        step = 0
    else:
        snapshot, step = get_snapshot(FLAGS.model_folder)

    image_list = os.path.join(FLAGS.input_folder, 'image_list.txt')
    label_list = os.path.join(FLAGS.input_folder, 'label_list.txt')

    train(image_list, label_list, FLAGS.model_folder,
          snapshot, step, FLAGS.debug)


if __name__ == '__main__':
    tf.app.run()
