import os, glob, re
import tensorflow as tf

def sorted_ls_by_num(list):
    nums = []
    error_idx = len(list)
    for file in list:
        baldname = os.path.split(file)[1]
        baldname = os.path.splitext(baldname)[0]
        elements = re.findall(r'\d+', baldname)
        if len(elements) == 0:
            print('No numbers are found in %s' % file)
            nums.append(error_idx)
            error_idx += 1
        else:
            nums.append(int(elements[-1]))
    sorted_index = sorted(range(len(nums)), key = lambda k : nums[k])
    sorted_list = []
    for index in sorted_index:
        sorted_list.append(list[index])
    return sorted_list

def get_snapshot(dir):
    os.chdir(dir)
    files = glob.glob("model.ckpt-*.index")
    if len(files) == 0:
        return None, 0
    else:
        files = sorted_ls_by_num(files)
        snapshot = files[-1]
        snapshot = os.path.splitext(snapshot)[0]
        step = int(snapshot.split('-')[-1])
        snapshot = os.path.join(dir, snapshot)
        return snapshot, step

def get_num_trainable_params():
    total_parameters = 0
    for variable in tf.trainable_variables():
        shape = variable.get_shape()
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim.value
        total_parameters += variable_parameters
    return total_parameters

def get_num_flops(sess):
    run_meta = tf.RunMetadata()
    opts = tf.profiler.ProfileOptionBuilder.float_operation()
    flops = tf.profiler.profile(sess.graph, run_meta=run_meta, cmd='op', options=opts)

    print(opts)
    opts = tf.profiler.ProfileOptionBuilder.trainable_variables_parameter()
    params = tf.profiler.profile(sess.graph, run_meta=run_meta, cmd='op', options=opts)
    print("{:,} --- {:,}".format(flops.total_float_ops, params.total_parameters))
    exit()