from cnn_wrapper.SCoordNet import SCoordNet
from cnn_wrapper.OFlowNet import OFlowNet
from util import *
from tools.util import bilinear_sampler

class KFNetDataSpec():
    def __init__(self,
                 batch_size=4,
                 image_size=(480, 640),
                 channels = 3,
                 crop_size=(480, 640),
                 focal_x=525.,
                 focal_y=525.,
                 u=320.,
                 v=240.,
                 scene='stairs'):
        self.batch_size = batch_size
        self.image_size = image_size
        self.channels = channels
        self.crop_size = crop_size
        self.focal_x = focal_x
        self.focal_y = focal_y
        self.u = u
        self.v = v
        self.scene = scene
        if self.scene == 'chess':
            self.image_num = 4000
            self.sequence_length = 1000
            self.num_sequence = 4
        elif self.scene == 'fire':
            self.image_num = 2000
            self.sequence_length = 1000
            self.num_sequence = 2
        elif self.scene == 'heads':
            self.image_num = 1000
            self.sequence_length = 1000
            self.num_sequence = 1
        elif self.scene == 'office':
            self.image_num = 6000
        elif self.scene == 'pumpkin':
            self.image_num = 4000
        elif self.scene == 'redkitchen':
            self.image_num = 7000
        elif self.scene == 'stairs':
            self.image_num = 2000
            self.sequence_length = 500
            self.num_sequence = 4

        self.stepvalue = self.image_num * 75 / self.batch_size  # drop the learning rate by half every 75 epochs
        self.max_steps = self.stepvalue * 5


class KFNet():
    def __init__(self, images, spec,
                 train_scoordnet, train_oflownet, dropout_rate=0.5, seed=None, reuse=tf.AUTO_REUSE):
        self.focal_x = spec.focal_x
        self.focal_y = spec.focal_y
        self.u = spec.u
        self.v = spec.v

        self.reuse = reuse
        self.train_scoordnet = train_scoordnet
        self.train_oflownet = train_oflownet
        self.seed = seed
        self.dropout_rate = dropout_rate

        self.batch_size = spec.batch_size
        self.height = spec.image_size[0]
        self.width = spec.image_size[1]
        self.flow_sample_rate = 8
        self.min_uncertainty = 1e-5

        self.images = images
        self.scoordnet = self.BuildSCoordNet()
        self.temp_feat_maps = self.BuildOFlowFeat()


    ####################### I/O #######################
    def GetInputImages(self):
        return self.images

    def GetMeasureCoord(self):
        return self.scoordnet.GetOutput()

    def GetKFCoordRecursive(self, last_coord, last_uncertainty):
        """
        Get coordinate map from kalman filter
        :param last_coord: 1xHxWx3
        :param last_uncertainty: 1xHxWx1
        :return: HxWx3
        """
        measure_coord, measure_uncertainty = self.GetMeasureCoord2()

        feat_map1 = tf.slice(self.temp_feat_maps, [0, 0, 0, 0], [1, -1, -1, -1])
        feat_map2 = tf.slice(self.temp_feat_maps, [1, 0, 0, 0], [1, -1, -1, -1])

        temp_coord, temp_uncertainty = self.BuildOFlowNet(feat_map1, feat_map2, last_coord, last_uncertainty)
        KF_coord, KF_uncertainty = self.BuildKFCoord(temp_coord, temp_uncertainty, measure_coord, measure_uncertainty)

        return temp_coord, temp_uncertainty, KF_coord, KF_uncertainty

    def GetKFCoordBatch(self):
        """
        Get coordinate map from kalman filter
        :return: HxWx3
        """
        pred_coord_map, pred_uncertainty_map = self.GetMeasureCoord()
        measure_coord1 = tf.slice(pred_coord_map, [0, 0, 0, 0], [1, -1, -1, -1])
        measure_coord2 = tf.slice(pred_coord_map, [1, 0, 0, 0], [1, -1, -1, -1])
        measure_coord3 = tf.slice(pred_coord_map, [2, 0, 0, 0], [1, -1, -1, -1])
        measure_coord4 = tf.slice(pred_coord_map, [3, 0, 0, 0], [1, -1, -1, -1])
        measure_uncertainty1 = tf.slice(pred_uncertainty_map, [0, 0, 0, 0], [1, -1, -1, -1])
        measure_uncertainty2 = tf.slice(pred_uncertainty_map, [1, 0, 0, 0], [1, -1, -1, -1])
        measure_uncertainty3 = tf.slice(pred_uncertainty_map, [2, 0, 0, 0], [1, -1, -1, -1])
        measure_uncertainty4 = tf.slice(pred_uncertainty_map, [3, 0, 0, 0], [1, -1, -1, -1])

        feat_map1 = tf.slice(self.temp_feat_maps, [0, 0, 0, 0], [1, -1, -1, -1])
        feat_map2 = tf.slice(self.temp_feat_maps, [1, 0, 0, 0], [1, -1, -1, -1])
        feat_map3 = tf.slice(self.temp_feat_maps, [2, 0, 0, 0], [1, -1, -1, -1])
        feat_map4 = tf.slice(self.temp_feat_maps, [3, 0, 0, 0], [1, -1, -1, -1])

        # frame1
        KF_coord1 = measure_coord1
        KF_uncertainty1 = measure_uncertainty1
        temp_coord1 = measure_coord1
        temp_uncertainty1 = measure_uncertainty1

        # frame2
        temp_coord2, temp_uncertainty2 = self.BuildOFlowNet(feat_map1, feat_map2, KF_coord1, KF_uncertainty1)
        KF_coord2, KF_uncertainty2 = self.BuildKFCoord(temp_coord2, temp_uncertainty2, measure_coord2, measure_uncertainty2)

        # frame3
        temp_coord3, temp_uncertainty3 = self.BuildOFlowNet(feat_map2, feat_map3, KF_coord2, KF_uncertainty2)
        KF_coord3, KF_uncertainty3 = self.BuildKFCoord(temp_coord3, temp_uncertainty3, measure_coord3, measure_uncertainty3)

        # frame4
        temp_coord4, temp_uncertainty4 = self.BuildOFlowNet(feat_map3, feat_map4, KF_coord3, KF_uncertainty3)
        KF_coord4, KF_uncertainty4 = self.BuildKFCoord(temp_coord4, temp_uncertainty4, measure_coord4, measure_uncertainty4)

        temp_coord = tf.concat([temp_coord1, temp_coord2, temp_coord3, temp_coord4], axis=0)
        temp_uncertainty = tf.concat([temp_uncertainty1, temp_uncertainty2, temp_uncertainty3, temp_uncertainty4], axis=0)

        KF_coord = tf.concat([KF_coord1, KF_coord2, KF_coord3, KF_coord4], axis=0)
        KF_uncertainty = tf.concat([KF_uncertainty1, KF_uncertainty2, KF_uncertainty3, KF_uncertainty4], axis=0)

        return temp_coord, temp_uncertainty, KF_coord, KF_uncertainty

    def BuildKFCoord(self, last_coord, last_uncertainty, measure_coord, measure_uncertainty):
        """
        Get coordinate map from kalman filter
        :return: HxWx3
        """
        with tf.variable_scope('KF'):
            last_variance = tf.square(last_uncertainty)
            measure_variance = tf.square(measure_uncertainty)
            weight_K = tf.div(last_variance, last_variance + measure_variance)
            weight_K_3 = tf.tile(weight_K, [1, 1, 1, 3])
            KF_coord = tf.maximum(1.0 - weight_K_3, 0.0) * last_coord + weight_K_3 * measure_coord
            KF_variance = tf.maximum(1.0 - weight_K, 0.0) * last_variance
            KF_uncertainty = tf.sqrt(KF_variance)

        return KF_coord, KF_uncertainty

    def GetInnovation(self, measure_coord_map, measure_uncertainty_map, temp_coord_map, temp_uncertainty_map):
        measure_variance = tf.square(measure_uncertainty_map)
        temp_variance = tf.square(temp_uncertainty_map)

        inno_mean = measure_coord_map - temp_coord_map
        inno_variance = temp_variance + measure_variance
        inno_uncertainty = tf.sqrt(inno_variance)

        return inno_mean, inno_uncertainty

    def GetNIS(self, measure_coord_map, measure_uncertainty_map, temp_coord_map, temp_uncertainty_map):
        """
        :return: BxHxWx3
        """
        with tf.variable_scope('NIS'):
            inno_mean, inno_uncertainty = self.GetInnovation(measure_coord_map, measure_uncertainty_map,
                                                         temp_coord_map, temp_uncertainty_map)
            inno_variance = tf.square(inno_uncertainty)
            inno_variance = tf.tile(inno_variance, [1, 1, 1, 3])
            NIS = tf.div(tf.square(inno_mean), inno_variance)
        return NIS  # 1xHxWx3
    ####################### eof I/O #######################


    ####################### loss function #######################
    def loss(self):
        return None

    def CoordLossWithUncertainty(self, pred_coord_map, uncertainty_map, gt_coord_map,
                                 mask=None, dist_threshold=0.05, transform=None, downsample=None):
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
        if transform is not None:
            pred_coord_map = ApplyTransform(pred_coord_map, transform)

        if downsample:
            gt_coord_map = tf.image.resize_nearest_neighbor(gt_coord_map, [height, width])
            if mask is not None:
                mask = tf.image.resize_nearest_neighbor(mask, [height, width])
                mask = tf.cast(tf.equal(mask, 1.0), tf.float32)

        diff_coord_map = tf.reduce_sum(tf.square(pred_coord_map - gt_coord_map), axis=-1, keepdims=True, name='diff_coord_map')  # BxHxWx1
        uncertainty_map = tf.maximum(uncertainty_map, self.min_uncertainty, name='uncertainty_map')
        loss_map = tf.add(3.0 * tf.log(uncertainty_map), diff_coord_map / (2. * tf.square(uncertainty_map)), name='loss_map')
        loss_map = tf.minimum(loss_map, -2.0)

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

    def MeasureCoordLoss(self, gt_coords, mask, transform=None, pixel_map=None, poses=None, images=None):
        """
        :param gt_coords:
        :param mask:
        :param transform: 4x4 Transform applied to predicetd coordinates
        :return:
        """
        pred_coord_map, pred_uncertainty_map = self.GetMeasureCoord()
        if transform is not None:
            pred_coord_map = ApplyTransform(pred_coord_map, transform)
        loss, accuracy = self.CoordLossWithUncertainty(pred_coord_map, pred_uncertainty_map, gt_coords, mask)

        if pixel_map is not None and poses is not None:
            reproj_loss = self.ReprojectionLoss(pred_coord_map, pixel_map, poses, mask)
            loss = loss + 50 * reproj_loss

        if images is not None:
            smooth_loss = self.SmoothLoss(pred_coord_map, images, mask)
            loss = loss + 50 * smooth_loss

        return loss, accuracy

    def TemporalCoordLoss(self, gt_coords, mask, transform=None, pixel_map=None, poses=None, images=None):
        """
        :param coords:
        :param mask:
        :param transform: 4x4 Transform applied to predicetd coordinates
        :return:
        """
        temp_coord_map, temp_uncertainty_map, _, _ = self.GetKFCoordBatch()

        if transform is not None:
            temp_coord_map = ApplyTransform(temp_coord_map, transform)

        loss, accuracy = self.CoordLossWithUncertainty(temp_coord_map, temp_uncertainty_map, gt_coords, mask)

        if pixel_map is not None and poses is not None:
            reproj_loss = self.ReprojectionLoss(temp_coord_map, pixel_map, poses, mask)
            loss = loss + 10 * reproj_loss

        if images is not None:
            smooth_loss = self.SmoothLoss(temp_coord_map, images, mask)
            loss = loss + 50 * smooth_loss

        return loss, accuracy

    def KFCoordLoss(self, gt_coords, mask, transform=None, pixel_map=None, poses=None, images=None):
        """
        :param gt_coords:
        :param mask:
        :param transform: 4x4 Transform applied to predicetd coordinates
        :return:
        """
        _, _, KF_coord_map, KF_uncertainty_map = self.GetKFCoordBatch()
        if transform is not None:
            KF_coord_map = ApplyTransform(KF_coord_map, transform)
        loss, accuracy = self.CoordLossWithUncertainty(KF_coord_map, KF_uncertainty_map, gt_coords, mask)

        if pixel_map is not None and poses is not None:
            reproj_loss = self.ReprojectionLoss(KF_coord_map, pixel_map, poses, mask)
            loss = loss + 10 * reproj_loss

        if images is not None:
            smooth_loss = self.SmoothLoss(KF_coord_map, images, mask)
            loss = loss + 50 * smooth_loss

        return loss, accuracy
    ####################### eof loss function #######################

    def BuildSCoordNet(self):
        with tf.variable_scope('ScoreNet'):
            return SCoordNet({'input': self.images},
                            is_training=self.train_scoordnet,
                            focal_x=self.focal_x,
                            focal_y=self.focal_y,
                            u=self.u,
                            v=self.v,
                            dropout_rate=self.dropout_rate,
                            seed=self.seed,
                            reuse=self.reuse)

    def BuildOFlowFeat(self):
        with tf.variable_scope('Temporal'):
            images = tf.multiply(tf.subtract(self.images, 128.0), 0.00625)
            feat_maps = tf.layers.conv2d(images, filters=16, kernel_size=3, strides=1, activation=tf.nn.relu,
                                         padding='SAME',
                                         trainable=self.train_oflownet, reuse=self.reuse, name='feat1')  # 640x480
            feat_maps = tf.layers.conv2d(feat_maps, filters=32, kernel_size=3, strides=2, activation=tf.nn.relu,
                                         padding='SAME',
                                         trainable=self.train_oflownet, reuse=self.reuse, name='feat2')  # 320x240
            feat_maps = tf.layers.conv2d(feat_maps, filters=32, kernel_size=3, strides=1, activation=tf.nn.relu,
                                         padding='SAME',
                                         trainable=self.train_oflownet, reuse=self.reuse, name='feat3')  # 320x240
            feat_maps = tf.layers.conv2d(feat_maps, filters=64, kernel_size=3, strides=2, activation=tf.nn.relu,
                                         padding='SAME',
                                         trainable=self.train_oflownet, reuse=self.reuse, name='feat4')  # 160x120
            feat_maps = tf.layers.conv2d(feat_maps, filters=64, kernel_size=3, strides=1, activation=tf.nn.relu,
                                         padding='SAME',
                                         trainable=self.train_oflownet, reuse=self.reuse, name='feat5')  # 160x120
            feat_maps = tf.layers.conv2d(feat_maps, filters=128, kernel_size=3, strides=2, activation=tf.nn.relu,
                                         padding='SAME',
                                         trainable=self.train_oflownet, reuse=self.reuse, name='feat6')  # 80x60
            feat_maps = tf.layers.conv2d(feat_maps, filters=32, kernel_size=3, strides=1, activation=None,
                                         padding='SAME',
                                         trainable=self.train_oflownet, reuse=self.reuse, name='feat7')  # 80x60

            feat_maps = tf.nn.l2_normalize(feat_maps, axis=-1)
        return feat_maps

    def BuildCoordVolume(self, feat_map1, feat_map2, window_size):
        diff_feats = []
        shift_offsets = []
        half_window_size = window_size // 2

        for i in range(window_size):  # y
            for j in range(window_size):  # x
                offset = [j - half_window_size, i - half_window_size]
                minus_offset = [-(j - half_window_size), -(i - half_window_size)]
                shift_feat_map1 = tf.contrib.image.translate(feat_map1, minus_offset, interpolation='NEAREST')
                diff_feat = feat_map2 - shift_feat_map1  # 1xHxWxC
                diff_feats.append(diff_feat)
                shift_offsets.append(tf.expand_dims(offset, axis=0))

        diff_feats = tf.concat(diff_feats, axis=-1, name='diff_feat')  # 1xHxWx300
        shift_offsets = tf.cast(tf.concat(shift_offsets, axis=0), tf.float32)  # 100x2
        return diff_feats, shift_offsets

    def BuildOFlowNet(self, feat_map1, feat_map2, coord_map1, uncertainty1):
        """
        infer temporal coord map2 from coord_map1
        :return: HxWx3, HxWx1
        """
        batch_size, height, width, channel = feat_map1.get_shape().as_list()

        # construct cost volume
        window_size = 64 // self.flow_sample_rate  # ensure a flow field 96x96 of original resolution
        window_area = window_size * window_size
        diff_feats, shift_offsets = self.BuildCoordVolume(feat_map1, feat_map2, window_size)
        diff_feats = tf.reshape(diff_feats, shape=[-1, window_size, window_size, channel])

        with tf.variable_scope('Temporal'):
            coord_flow_net = OFlowNet({'input': diff_feats},
                                          window_area,
                                          is_training=self.train_oflownet,
                                          reuse=self.reuse)  # BHWx1x1x100
            prob, transition_uncertainty = coord_flow_net.GetOutput()  # BHWx100, BHWx1

        prob_reshape = tf.reshape(prob, shape=[-1, 1, window_area], name='prob_reshape')  # Nx1x100
        shift_offsets = tf.tile(tf.expand_dims(shift_offsets, axis=0), [batch_size * height * width, 1, 1],
                                name='shift_offsets')  # Nx100x2
        flow = tf.matmul(prob_reshape, shift_offsets)  # Nx1x2
        flow = tf.reshape(flow, shape=[batch_size, height, width, 2], name='flow')  # BxHxWx2
        pixel_map = tf.add(GetPixelMap(batch_size, height, width), flow, name='last_pixel_map')  # BxHxWx2

        coord_map1 = tf.add(coord_map1, 0.0, name='coord_map1')
        temp_coord = bilinear_sampler(coord_map1, pixel_map)  # BxHxWx3
        temp_coord = tf.add(temp_coord, 0.0, name='temp_coord')
        last_uncertainty = bilinear_sampler(uncertainty1, pixel_map)  # BxHxWx1
        last_uncertainty = tf.add(last_uncertainty, 0.0, name='last_uncertainty')
        last_variance = tf.square(last_uncertainty)
        last_variance = tf.maximum(last_variance, self.min_uncertainty * self.min_uncertainty)

        transition_uncertainty = tf.reshape(transition_uncertainty, shape=[batch_size, height, width, 1])
        transition_variance = tf.square(transition_uncertainty)
        transition_variance = tf.maximum(transition_variance, self.min_uncertainty * self.min_uncertainty)

        temp_variance = transition_variance + last_variance
        temp_uncertainty = tf.sqrt(temp_variance, name='temp_uncertainty')

        return temp_coord, temp_uncertainty

    def ReprojectionLoss(self, coord_map, pixel_map, poses, mask=None):
        """
        :param coord_map: BxHxWx3
        :param pixel_map: BxHxWx2   The pixel map is normalized
        :param poses: Bx4x4     Transform from world frame to camera frame
        :param mask: BxHxWx1
        :return:
        """
        batch_size, height, width, _ = coord_map.get_shape().as_list()
        camera_coord_map = ApplyTransform(coord_map, poses)  # BxHxWx3
        homo_pixel_map = HomoCoord(pixel_map)  # BxHxWx3

        camera_coord_map = tf.nn.l2_normalize(camera_coord_map, axis=-1)
        homo_pixel_map = tf.nn.l2_normalize(homo_pixel_map, axis=-1)
        error_map = tf.square(camera_coord_map - homo_pixel_map)  # distance between normalized vectors

        if mask is not None:
            valid_pixel = tf.reduce_sum(mask) + 1.
            error_map = mask * error_map
        else:
            valid_pixel = batch_size * height * width

        error = tf.reduce_sum(error_map) / (valid_pixel + 1)
        return error

    def SmoothLoss(self, coord_map, img, mask=None):
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

        disp_gx, disp_gy = _gradient(coord_map)  # Bx(H-1)x(W-1)xC
        img_gx, img_gy = _gradient(img)  # Bx(H-1)x(W-1)x3

        disp_gx = tf.reduce_mean(tf.square(disp_gx), axis=3, keepdims=True)  # Bx(H-1)x(W-1)x1
        disp_gy = tf.reduce_mean(tf.square(disp_gy), axis=3, keepdims=True)

        scale = 0.625
        weights_x = tf.exp(-scale * tf.reduce_mean(tf.abs(img_gx), axis=3, keepdims=True))  # Bx(H-1)x(W-1)x1
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

############## These functions are used for recursive inference via Kalman filter ##############
    def GetMeasureCoord2(self):
        coord_map, uncertainty_map = self.GetMeasureCoord()
        coord_map2 = tf.slice(coord_map, [1, 0, 0, 0], [1, -1, -1, -1])
        uncertainty_map2 = tf.slice(uncertainty_map, [1, 0, 0, 0], [1, -1, -1, -1])
        return coord_map2, uncertainty_map2

    def GetTemporalCoord2(self, coord_map1, uncertainty1):
        """
        :param coord_map1: 1xHxWx3 - last estimation
        :param uncertainty1: 1xHxWx1 - last estimation
        :return:
        """
        # extract features
        feat_map1 = tf.slice(self.temp_feat_maps, [0, 0, 0, 0], [1, -1, -1, -1])
        feat_map2 = tf.slice(self.temp_feat_maps, [1, 0, 0, 0], [1, -1, -1, -1])

        temp_coord2, temp_uncertainty2 = self.BuildOFlowNet(feat_map1, feat_map2, coord_map1, uncertainty1)
        return temp_coord2, temp_uncertainty2

    def GetKFCoord2(self, KF_coord_map1, KF_uncertainty1):
        measure_coord_map2, measure_uncertainty_map2 = self.GetMeasureCoord2()
        measure_variance2 = tf.square(measure_uncertainty_map2)

        temp_coord_map2, temp_uncertainty_map2 = self.GetTemporalCoord2(KF_coord_map1, KF_uncertainty1)
        temp_variance2 = tf.square(temp_uncertainty_map2)

        variance_12 = temp_variance2
        weight_K = tf.div(variance_12, variance_12 + measure_variance2)
        weight_K_3 = tf.tile(weight_K, [1, 1, 1, 3])
        KF_coord_map2 = tf.maximum(1.0 - weight_K_3, 0.0) * temp_coord_map2 + weight_K_3 * measure_coord_map2
        KF_variance2 = tf.square(1.0 - weight_K) * variance_12 + tf.square(weight_K) * measure_variance2
        KF_uncertainty2 = tf.sqrt(KF_variance2)
        return KF_coord_map2, KF_uncertainty2
############## These functions are used for recursive inference via Kalman filter ##############



