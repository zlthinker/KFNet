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

def read_lines(filepath):
    with open(filepath) as fin:
        lines = fin.readlines()
    lines = [line.strip() for line in lines]
    return lines