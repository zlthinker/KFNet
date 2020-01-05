import tensorflow as tf
from util import *


def SmoothLoss(disp, img, mask=None):
    """
    Smooth loss is defined to ensure smoothness between neighboring pixels,
    while weighted by image gradients to mitigate the effect of discontinuity
    caused by boundary
    :param disp: BxHxWxC
    :param img: BxHxWx3 or BxHxWx1, 0~255
    :param mask: BxHxWx1
    :return:
    """
    def _gradient(img):
        gx = img[:, :, :-1, :] - img[:, :, 1:, :]
        gy = img[:, :-1, :, :] - img[:, 1:, :, :]
        return gx, gy

    disp_gx, disp_gy = _gradient(disp)      # Bx(H-1)x(W-1)xC
    img_gx, img_gy = _gradient(img)         # Bx(H-1)x(W-1)x3

    disp_gx = tf.reduce_mean(tf.square(disp_gx), axis=3, keepdims=True)        # Bx(H-1)x(W-1)x1
    disp_gy = tf.reduce_mean(tf.square(disp_gy), axis=3, keepdims=True)

    scale = 0.625
    weights_x = tf.exp(-scale * tf.reduce_mean(tf.abs(img_gx), axis=3, keepdims=True)) # Bx(H-1)x(W-1)x1
    weights_y = tf.exp(-scale * tf.reduce_mean(tf.abs(img_gy), axis=3, keepdims=True))

    smoothness_x = disp_gx * weights_x
    smoothness_y = disp_gy * weights_y
    if mask is not None:
        mask_x = mask[:, :, :-1, :]
        mask_y = mask[:, :-1, :, :]
        smoothness_x = smoothness_x * mask_x
        smoothness_y = smoothness_y * mask_y
        valid_pixel = tf.reduce_sum(mask) + 1.
        loss = (tf.reduce_sum(smoothness_x) + tf.reduce_sum(smoothness_y)) / valid_pixel
    else:
        loss = tf.reduce_mean(smoothness_x) + tf.reduce_mean(smoothness_y)
    return loss

def CoordLoss(pred_coord_map, gt_coord_map, mask=None, dist_threshold=0.05):
    """
    L2 distance between predicted coordinates and groundtruth coordinates
    :param pred_coord_map: BxHxWxC
    :param gt_coord_map: BxHxWxC
    :return:
    """
    shape = pred_coord_map.get_shape().as_list()
    batch_size = shape[0]
    height = shape[1]
    width = shape[2]

    diff_coord_map = tf.reduce_sum(tf.square(pred_coord_map - gt_coord_map), axis=-1, keepdims=True)
    if mask is not None:
        valid_pixel = tf.reduce_sum(mask) + 1.
        diff_coord_map = mask * diff_coord_map
    else:
        valid_pixel = batch_size * height * width

    loss = tf.reduce_sum(diff_coord_map) / valid_pixel
    thres_coord_map = tf.maximum(diff_coord_map - dist_threshold * dist_threshold, 0)
    with tf.device('/device:CPU:0'):
        num_accurate = valid_pixel - tf.cast(tf.count_nonzero(thres_coord_map), tf.float32)
    accuracy = num_accurate / valid_pixel

    return loss, accuracy


def CoordLossWithUncertainty(pred_coord_map, uncertainty_map, gt_coord_map, mask=None, dist_threshold=0.05):
    """
    loss = t + (x-mean)^2 / 2 exp(2t)
    :param pred_coord_map: BxHxWxC
    :param uncertainty_map: BxHxWx1
    :param gt_coord_map: BxHxWxC
    :return:
    """
    shape = pred_coord_map.get_shape().as_list()
    batch_size = shape[0]
    height = shape[1]
    width = shape[2]
    diff_coord_map = tf.reduce_sum(tf.square(pred_coord_map - gt_coord_map), axis=-1, keepdims=True,
                                   name='diff_coord_map')  # BxHxWx1
    uncertainty_map = tf.maximum(uncertainty_map, 1e-5, name='uncertainty_map')
    loss_map = tf.add(3.0 * tf.log(uncertainty_map), diff_coord_map / (2. * tf.square(uncertainty_map)),
                      name='loss_map')

    if mask is not None:
        valid_pixel = tf.reduce_sum(mask) + 1.
        diff_coord_map = mask * diff_coord_map
        loss_map = mask * loss_map
    else:
        valid_pixel = batch_size * height * width

    loss = tf.reduce_sum(loss_map) / valid_pixel
    thres_coord_map = tf.maximum(diff_coord_map - dist_threshold * dist_threshold, 0)
    with tf.device('/device:CPU:0'):
        num_accurate = valid_pixel - tf.cast(tf.count_nonzero(thres_coord_map), tf.float32)
    accuracy = num_accurate / valid_pixel

    return loss, accuracy


def EPnPLoss(coord_map, pixel_map, control_points, spec, mask=None):
    """
    :param coord_map: BxHxWx3, barycentric coordinates
    :param pixel_map: BxHxWx2
    :param control_points: Bx12
    :return:
    """
    coefficients1 = get_EPnP_coefficient1(coord_map, pixel_map, spec)  # BxHxWx12
    coefficients2 = get_EPnP_coefficient2(coord_map, pixel_map, spec)  # BxHxWx12

    if mask is not None:
        valid_pixel = tf.reduce_sum(mask) + 1.
        mask = tf.tile(mask, [1, 1, 1, 12])
        coefficients1 = mask * coefficients1
        coefficients2 = mask * coefficients2
    else:
        valid_pixel = spec.batch_size * spec.crop_size[0] * spec.crop_size[1]

    coefficients1 = tf.reshape(coefficients1, shape=[spec.batch_size, -1, 12], name='coeff1')  # BxHWx12
    coefficients2 = tf.reshape(coefficients2, shape=[spec.batch_size, -1, 12], name='coeff2')  # BxHWx12

    control_points = tf.expand_dims(control_points, axis=-1)  # Bx12x1
    pnp_mul1 = tf.matmul(coefficients1, control_points, name='pnp_error_vector1')  # BxHWx1
    pnp_mul2 = tf.matmul(coefficients2, control_points, name='pnp_error_vector2')  # BxHWx1
    pnp_mul1_square = tf.square(pnp_mul1)
    pnp_mul2_square = tf.square(pnp_mul2)
    loss = tf.reduce_sum(pnp_mul1_square + pnp_mul2_square) / valid_pixel
    return loss

def WeightedEPnPLoss(coord_map, pixel_map, control_points, weight_map, spec, mask=None):
    """
    :param coord_map: BxHxWx3, barycentric coordinates
    :param pixel_map: BxHxWx2
    :param control_points: Bx12
    :param weight_map: BxHxWx1 range[0, 1]
    :return:
    """
    coefficients1 = get_EPnP_coefficient1(coord_map, pixel_map, spec)  # BxHxWx12
    coefficients2 = get_EPnP_coefficient2(coord_map, pixel_map, spec)  # BxHxWx12

    if mask is not None:
        valid_pixel = tf.reduce_sum(mask) + 1.
        weight_map = mask * weight_map
        mask_12 = tf.tile(mask, [1, 1, 1, 12])
        coefficients1 = mask_12 * coefficients1
        coefficients2 = mask_12 * coefficients2
    else:
        valid_pixel = spec.batch_size * spec.crop_size[0] * spec.crop_size[1]

    coefficients1 = tf.reshape(coefficients1, shape=[spec.batch_size, -1, 12], name='coeff1')  # BxHWx12
    coefficients2 = tf.reshape(coefficients2, shape=[spec.batch_size, -1, 12], name='coeff2')  # BxHWx12

    weight_map_12 = tf.tile(tf.reshape(weight_map, [spec.batch_size, -1, 1]), [1, 1, 12])  # BxHWx12
    coefficients1 = coefficients1 * weight_map_12
    coefficients2 = coefficients2 * weight_map_12

    control_points = tf.expand_dims(control_points, axis=-1)  # Bx12x1
    pnp_mul1 = tf.matmul(coefficients1, control_points, name='pnp_error_vector1')  # BxHWx1
    pnp_mul2 = tf.matmul(coefficients2, control_points, name='pnp_error_vector2')  # BxHWx1
    pnp_mul1_square = tf.square(pnp_mul1)
    pnp_mul2_square = tf.square(pnp_mul2)
    pnp_loss = tf.reduce_sum(pnp_mul1_square + pnp_mul2_square) / valid_pixel

    # regular
    weight_map_inverse = mask * (1.0 - weight_map)
    regular_loss = tf.reduce_sum(weight_map_inverse) / valid_pixel

    loss = pnp_loss + 5e-4 * regular_loss

    return loss, pnp_loss, regular_loss

def ReprojectionLoss(coord_map, pixel_map, poses, mask=None):
    """
    :param coord_map: BxHxWx3
    :param pixel_map: BxHxWx2   The pixel map is normalized
    :param poses: Bx4x4     Transform from world frame to camera frame
    :param mask: BxHxWx1
    :return:
    """
    batch_size, height, width, _ = coord_map.get_shape().as_list()
    camera_coord_map = ApplyTransform(coord_map, poses) # BxHxWx3
    homo_pixel_map = HomoCoord(pixel_map)               # BxHxWx3

    camera_coord_map = tf.nn.l2_normalize(camera_coord_map, axis=-1)
    homo_pixel_map = tf.nn.l2_normalize(homo_pixel_map, axis=-1)
    error_map = tf.square(camera_coord_map - homo_pixel_map) # distance between normalized vectors

    if mask is not None:
        valid_pixel = tf.reduce_sum(mask) + 1.
        error_map = mask * error_map
    else:
        valid_pixel = batch_size * height * width

    error = tf.reduce_sum(error_map) / valid_pixel
    return error








