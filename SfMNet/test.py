import sys
sys.path.append('/run/media/larry/fafb882a-0878-4e0a-9ccb-2fb979b7f717/e3dengine/tfmatch')
import tensorflow as tf
from tools.io import read_lines
import numpy as np
from SfMNet import *
from tools.common import Notify
from tools.io import get_snapshot
import time, os, argparse
from tensorflow.python import debug as tf_debug
import matplotlib.pyplot as plt
from train import *


def TestOpticalFlow(image_list, pose_list, flownet_snapshot):
    spec = SfMNetDataSpec()
    image_paths = read_lines(image_list)
    images = []
    indexes = [0, 1, 2, 3]
    for id in indexes:
        image_path = image_paths[id]
        print image_path
        image = tf.image.decode_png(tf.read_file(image_path), channels=spec.channels)
        image.set_shape((spec.image_size[0], spec.image_size[1], spec.channels))
        image = tf.cast(image, tf.float32)
        image = tf.image.crop_to_bounding_box(image, 0, 0, spec.crop_size[0], spec.crop_size[1])
        images.append(image)
    images = tf.stack(images, axis=0)  # BxHxWx3

    poses = tf.decode_raw(tf.read_file(pose_list), tf.float32)
    poses = tf.reshape(poses, [-1, 4, 4], name='poses')
    poses = tf.gather(poses, indexes)   # Bx4x4

    pixel_map = get_pixel_map(spec.crop_size[0], spec.crop_size[1])
    pixel_map = tf.expand_dims(pixel_map, dim=0)
    pixel_map = tf.tile(pixel_map, [len(indexes), 1, 1, 1]) # BxHxWx3

    sfmnet = SfMNet(images, pixel_map, poses, spec.focal_x, spec.focal_y, spec.u, spec.v, is_training=True, reuse=False)
    reproj_loss, warp_loss = sfmnet.loss()
    flow = sfmnet.GetOpticalFlow()  # BxBxHxWx2
    flow = tf.slice(flow, [0, 2, 0, 0, 0], [1, 1, -1, -1, -1])  # 1x1xHxWx2
    flow = tf.squeeze(flow) # HxWx2

    img1 = tf.squeeze(tf.slice(images, [0, 0, 0, 0], [1, -1, -1, -1])) / 255.0  # HxWx3
    img2 = tf.squeeze(tf.slice(images, [1, 0, 0, 0], [1, -1, -1, -1])) / 255.0  # HxWx3

    with tf.Session() as sess:
        RestoreFromScope(sess, flownet_snapshot, 'FlowNetS')

        out_warp_loss, out_img1, out_img2, out_flow = sess.run([warp_loss, img1, img2, flow])
        print out_warp_loss
        img_concat = np.concatenate([out_img1, out_img2], axis=1)
        flow_concat = np.concatenate([out_flow[:, :, 0], out_flow[:, :, 1]], axis=1)
        fig, axarr = plt.subplots(2, 1)
        plt.subplot(2, 1, 1)
        plt.imshow(img_concat)
        plt.subplot(2, 1, 2)
        plt.imshow(flow_concat)
        plt.show()

def TestReprojection(image_list, pose_list, coord_list, flow_file, flownet_snapshot):

    spec = SfMNetDataSpec()
    image_paths = read_lines(image_list)
    coord_paths = read_lines(coord_list)
    images = []
    coords = []
    masks = []
    indexes = [10, 20, 0]
    for id in indexes:
        image_path = image_paths[id]
        print image_path
        image = tf.image.decode_png(tf.read_file(image_path), channels=spec.channels)
        image.set_shape((spec.image_size[0], spec.image_size[1], spec.channels))
        image = tf.cast(image, tf.float32)
        image = tf.image.crop_to_bounding_box(image, 0, 0, spec.crop_size[0], spec.crop_size[1])
        images.append(image)

        coord_path = coord_paths[id]
        print coord_path
        coord_mask = tf.reshape(tf.decode_raw(tf.read_file(coord_path), tf.float32), [spec.image_size[0], spec.image_size[1], 4])
        coord = tf.slice(coord_mask, [0, 0, 0], [-1, -1, 3])
        coord = tf.image.crop_to_bounding_box(coord, 0, 0, spec.crop_size[0], spec.crop_size[1])
        coords.append(coord)

        mask = tf.slice(coord_mask, [0, 0, 3], [-1, -1, 1])
        mask = tf.image.crop_to_bounding_box(mask, 0, 0, spec.crop_size[0], spec.crop_size[1])
        masks.append(mask)

    images = tf.stack(images, axis=0)  # BxHxWx3
    coords = tf.stack(coords, axis=0)  # BxHxWx3
    masks = tf.stack(masks, axis=0)  # BxHxWx1

    poses = tf.decode_raw(tf.read_file(pose_list), tf.float32)
    poses = tf.reshape(poses, [-1, 4, 4], name='poses')
    poses = tf.gather(poses, indexes)   # Bx4x4

    pixel_map = get_pixel_map(spec.crop_size[0], spec.crop_size[1])
    pixel_map = tf.expand_dims(pixel_map, dim=0)
    pixel_map = tf.tile(pixel_map, [len(indexes), 1, 1, 1]) # BxHxWx3

    sfmnet = SfMNet(images, pixel_map, poses, spec.focal_x, spec.focal_y, spec.u, spec.v, is_training=True, reuse=False)
    flow_pixels = sfmnet.PixelWarp()    # BxBxHxWx2
    reproj_pixels = sfmnet.World2Pixel(coords, poses)   # BxBxHxWx2

    mask = tf.slice(masks, [2, 0, 0, 0], [1, -1, -1, -1])   # 1xHxWx1
    mask = tf.squeeze(tf.tile(mask, [1, 1, 1, 2]))  # 1xHxWx1

    gt_flow = tf.reshape(tf.decode_raw(tf.read_file(flow_file), tf.float32), [spec.image_size[0], spec.image_size[1], 2])
    gt_flow = tf.image.crop_to_bounding_box(gt_flow, 0, 0, spec.crop_size[0], spec.crop_size[1])
    flow_pixels = get_pixel_map(spec.crop_size[0], spec.crop_size[1]) + gt_flow
    flow_pixels = mask * flow_pixels  # HxWx2
    reproj_pixels = mask * tf.squeeze(tf.slice(reproj_pixels, [2, 1, 0, 0, 0], [1, 1, -1, -1, -1]))  # HxWx2
    error_square = tf.abs(flow_pixels - reproj_pixels, name='reproj_error_square')
    loss = tf.reduce_mean(error_square)

    img1 = tf.squeeze(tf.slice(images, [2, 0, 0, 0], [1, -1, -1, -1])) / 255.0  # HxWx3
    img2 = tf.squeeze(tf.slice(images, [1, 0, 0, 0], [1, -1, -1, -1])) / 255.0  # HxWx3

    with tf.Session() as sess:
        RestoreFromScope(sess, flownet_snapshot, 'FlowNetS')

        out_img1, out_img2, out_flow, out_reproj, out_loss = sess.run([img1, img2, flow_pixels, reproj_pixels, loss])
        print 'loss', out_loss

        img_concat = np.concatenate([out_img1, out_img2], axis=1)
        flow_concat = np.concatenate([out_flow[:, :, 0], out_flow[:, :, 1]], axis=1)
        reproj_concat = np.concatenate([out_reproj[:, :, 0], out_reproj[:, :, 1]], axis=1)
        reproj_concat = np.clip(reproj_concat, 0, 600)

        fig, axarr = plt.subplots(4, 1)
        plt.subplot(4, 1, 1)
        plt.imshow(img_concat)
        plt.subplot(4, 1, 2)
        plt.imshow(flow_concat)
        plt.subplot(4, 1, 3)
        plt.imshow(reproj_concat)
        plt.subplot(4, 1, 4)
        plt.imshow(flow_concat - reproj_concat)
        plt.show()

# image_list = sys.argv[1]
# pose_list = sys.argv[2]
# flownet_snapshot = sys.argv[3]
# coord_list = sys.argv[4]
# flow_file = sys.argv[5]
# """
# python SfMNet/test.py '/run/media/larry/fafb882a-0878-4e0a-9ccb-2fb979b7f717/e3dengine/tensorflow/SfMNet/chess/train/image_list.txt' '/run/media/larry/fafb882a-0878-4e0a-9ccb-2fb979b7f717/e3dengine/tensorflow/SfMNet/chess/train/pose_mat.bin'  '/run/media/larry/fafb882a-0878-4e0a-9ccb-2fb979b7f717/e3dengine/tensorflow/FlowNet/train/model/FlowNetS/rename/model.ckpt-1200000' '/run/media/larry/fafb882a-0878-4e0a-9ccb-2fb979b7f717/e3dengine/tensorflow/ScoreNet/frame/chess/train/label_list.txt' '/run/media/larry/fafb882a-0878-4e0a-9ccb-2fb979b7f717/e3dengine/tensorflow/FlowNet/train/test/0-20/gt_flow.bin'
# """
# TestReprojection(image_list, pose_list, coord_list, flow_file, flownet_snapshot)


def TestReprojectionEPnP(image_list, cp_list, coord_list, flow_file, flownet_snapshot):

    spec = SfMNetDataSpec()
    image_paths = read_lines(image_list)
    coord_paths = read_lines(coord_list)
    images = []
    coords = []
    masks = []
    indexes = [0, 10, 20]
    for id in indexes:
        image_path = image_paths[id]
        print image_path
        image = tf.image.decode_png(tf.read_file(image_path), channels=spec.channels)
        image.set_shape((spec.image_size[0], spec.image_size[1], spec.channels))
        image = tf.cast(image, tf.float32)
        image = tf.image.crop_to_bounding_box(image, 0, 0, spec.crop_size[0], spec.crop_size[1])
        images.append(image)

        coord_path = coord_paths[id]
        print coord_path
        coord_mask = tf.reshape(tf.decode_raw(tf.read_file(coord_path), tf.float32), [spec.image_size[0], spec.image_size[1], 4])
        coord = tf.slice(coord_mask, [0, 0, 0], [-1, -1, 3])
        coord = tf.image.crop_to_bounding_box(coord, 0, 0, spec.crop_size[0], spec.crop_size[1])
        coords.append(coord)

        mask = tf.slice(coord_mask, [0, 0, 3], [-1, -1, 1])
        mask = tf.image.crop_to_bounding_box(mask, 0, 0, spec.crop_size[0], spec.crop_size[1])
        masks.append(mask)

    images = tf.stack(images, axis=0)  # BxHxWx3
    coords = tf.stack(coords, axis=0)  # BxHxWx3
    masks = tf.stack(masks, axis=0)  # BxHxWx1

    cps = tf.decode_raw(tf.read_file(cp_list), tf.float32)
    cps = tf.reshape(cps, [-1, 12], name='control_points')
    cps = tf.gather(cps, indexes)   # Bx12

    pixel_map = get_pixel_map(spec.crop_size[0], spec.crop_size[1])
    pixel_map = tf.expand_dims(pixel_map, dim=0)
    pixel_map = tf.tile(pixel_map, [len(indexes), 1, 1, 1]) # BxHxWx2
    mask = tf.slice(masks, [0, 0, 0, 0], [1, -1, -1, -1])

    sfmnet = SfMNet(images, pixel_map, cps, spec.focal_x, spec.focal_y, spec.u, spec.v, is_training=True, reuse=False)
    gt_flow = tf.reshape(tf.decode_raw(tf.read_file(flow_file), tf.float32),
                         [spec.image_size[0], spec.image_size[1], 2])
    gt_flow = tf.image.crop_to_bounding_box(gt_flow, 0, 0, spec.crop_size[0], spec.crop_size[1])
    flow_pixels = get_pixel_map(spec.crop_size[0], spec.crop_size[1]) + gt_flow
    flow_pixels = tf.expand_dims(flow_pixels, axis=0)
    flow_pixels = sfmnet.NormalizePixelMap(flow_pixels) # 1xHxWx2


    coord = tf.slice(coords, [0, 0, 0, 0], [1, -1, -1, -1]) #1xHxWx3
    control_points = tf.slice(cps, [2, 0], [1, -1])  # 1x12
    loss = sfmnet.scorenet.EPnPLoss(coord, flow_pixels, control_points, mask)

    with tf.Session() as sess:
        RestoreFromScope(sess, flownet_snapshot, 'FlowNetS')

        out_loss = sess.run([loss])
        print 'loss: ', out_loss

image_list = sys.argv[1]
cp_list = sys.argv[2]
flownet_snapshot = sys.argv[3]
coord_list = sys.argv[4]
flow_file = sys.argv[5]
TestReprojectionEPnP(image_list, cp_list, coord_list, flow_file, flownet_snapshot)