#!/usr/bin/env python
"""
Copyright 2017, Zixin Luo, HKUST.
IO tools.
"""

from __future__ import print_function

import os, glob, re
import struct
from math import ceil, isnan
from struct import unpack
from random import shuffle

import numpy as np
import tensorflow as tf
from common import Notify, read_list


def hash_int_pair(ind1, ind2):
    '''
    Hash an Int pair
    :param ind1: int1
    :param ind2: int2
    :return: the hash index
    '''
    return ind1 * 2147483647 + ind2


def read_feature_repo(file_path):
    """Read feature file (*.sift)."""
    with open(file_path, 'rb') as fin:
        data = fin.read()

    head_length = 20
    head = data[0:head_length]
    feature_name, _, num_features, loc_dim, des_dim = struct.unpack('5i', head)
    keypts_length = loc_dim * num_features * 4

    if feature_name == ord('S') + (ord('I') << 8) + (ord('F') << 16) + (ord('T') << 24):
        print(Notify.INFO, 'Reading SIFT file',
              file_path, '#', num_features, Notify.ENDC)
        desc_length = des_dim * num_features
        desc_type = 'B'
    elif feature_name == 21384864:  # L2Net
        print(Notify.INFO, 'Reading L2NET file',
              file_path, '#', num_features, Notify.ENDC)
    else:
        print(Notify.FAIL, 'Unknown feature type.', Notify.ENDC)
        desc_length = des_dim * num_features * 4
        desc_type = 'f'

    keypts_data = data[head_length: head_length + keypts_length]
    keypts = np.array(struct.unpack('f' * loc_dim * num_features, keypts_data))
    keypts = np.reshape(keypts, (num_features, loc_dim))

    desc_data = data[head_length +
                     keypts_length: head_length + keypts_length + desc_length]
    desc = np.array(struct.unpack(
        desc_type * des_dim * num_features, desc_data))
    desc = np.reshape(desc, (num_features, des_dim))

    return keypts, desc


def read_mask_binary(bin_path, size):
    mask_dict = {}
    mask_size = size * size * 2
    record_size = 8 + mask_size
    if os.path.exists(bin_path):
        with open(bin_path, 'rb') as fin:
            data = fin.read()
        for i in range(0, len(data), record_size):
            decoded = unpack(
                'Q' + '?' * mask_size, data[i: i + record_size])
            mask_dict[decoded[0]] = decoded[1:]
        return mask_dict
    else:
        print(Notify.WARNING, 'Not exist', bin_path, Notify.ENDC)
        return None


def read_mask_to_bool(mask_file):
    mask_list = read_list(mask_file)
    mask_p1 = np.array(map(int, mask_list[0].split(',')), np.bool)
    mask_p2 = np.array(map(int, mask_list[1].split(',')), np.bool)
    return np.concatenate((mask_p1, mask_p2))


def read_mask(idx, size, imageset):
    """Read mask file."""
    mask_file = os.path.join(imageset, 'my_output',
                             'parsed_mask', str(idx[0]) + '_' + str(idx[1]) + '.jpg.mask')
    if not os.path.exists(mask_file):
        print(Notify.WARNING, 'Not exist', mask_file, Notify.ENDC)
        return None, None
    mask_list = read_list(mask_file)
    mask_p1 = np.array(map(int, mask_list[0].split(',')), np.int32)
    mask_p2 = np.array(map(int, mask_list[1].split(',')), np.int32)
    mask_p1 = np.reshape(mask_p1, size)
    mask_p2 = np.reshape(mask_p2, size)
    return mask_p1, mask_p2


def read_matching_pair(list_path, reproj_tol=4, max_num=50000):
    """Read matching pair"""
    content = open(list_path).read().splitlines()
    shuffle(content)
    pairs = []
    n_filter = 0
    for idx, val in enumerate(content):
        if idx > max_num:
            break
        strs = val.split(' ')
        offset_x1 = float(strs[1]) - float(strs[3])
        offset_y1 = float(strs[2]) - float(strs[4])
        offset_x2 = float(strs[10]) - float(strs[12])
        offset_y2 = float(strs[11]) - float(strs[13])
        if offset_x1 > reproj_tol or \
                offset_x2 > reproj_tol or \
                offset_y1 > reproj_tol or \
                offset_y2 > reproj_tol:
            n_filter += 1
            continue
        pairs.append(strs[0] + ' ' + strs[9])
    return pairs, n_filter


def read_local_match(file_path):
    """Read local match file"""
    matches = []
    with open(file_path, 'rb') as fin:
        while True:
            rin = fin.read(24)
            if rin == '':
                # EOF
                break
            idx0, idx1, num = struct.unpack('L' * 3, rin)
            bytes_theta = num * 52
            corr = np.fromstring(fin.read(bytes_theta),
                                 dtype=np.float32).reshape(-1, 13)
            matches.append([idx0, idx1, corr])
    return matches


def split_to_samples(input_local_match, num_corr, out_root, offset=0):
    """Split the local match file to multiple training samples."""
    if not os.path.exists(out_root):
        os.mkdir(out_root)
    else:
        _ = [os.remove(os.path.join(out_root, tmp_sample))
             for tmp_sample in os.listdir(out_root)]
    matches = read_local_match(input_local_match)
    shuffle(matches)
    for idx, val in enumerate(matches):
        img_idx = [val[0] + offset, val[1] + offset]
        corr = val[2]
        np.random.shuffle(corr)
        num_samples = int(ceil(val[2].shape[0] / num_corr))
        for i in range(num_samples):
            head = struct.pack('f' * 2, *img_idx)
            theta0 = corr[i * num_corr: (i + 1) * num_corr, 0:6].flatten()
            theta1 = corr[i * num_corr: (i + 1) * num_corr, 6:12].flatten()
            geo_dis = corr[i * num_corr: (i + 1) * num_corr, 12]
            geo_dis = [1. if isnan(i_) else i_ for i_ in geo_dis]
            ave_geo_dis = sum(geo_dis) / len(geo_dis)
            theta = np.concatenate((theta0, theta1, geo_dis))
            with open(os.path.join(out_root, '_'.join([str(idx), str(i), "%.2f" % ave_geo_dis]) + '.sample'), 'wb') as fout:
                fout.write(head)
                fout.write(struct.pack('f' * num_corr * 13, *theta))

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