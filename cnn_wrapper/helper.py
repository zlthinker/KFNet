#!/usr/bin/env python
"""
Copyright 2017, Zixin Luo, HKUST.
Network specifications.
"""

from cnn_wrapper.half_googlenet import Half_GoogleNet
from cnn_wrapper.googlenet import GoogleNet
from cnn_wrapper.l2net import L2Net
from cnn_wrapper.mvdesc import MVDesc
from cnn_wrapper.ScoreNet import ScoreNet
from cnn_wrapper.PoseNet import PoseNet
from cnn_wrapper.FlowNetS import FlowNetS
from cnn_wrapper.FlowNetSD import FlowNetSD
from cnn_wrapper.CoordFlowNet import CoordFlowNet
from pynvml import *



class DataSpec(object):
    """Input data specifications for an ImageNet model."""

    def __init__(self,
                 batch_size,
                 input_size,
                 scale=1.,
                 central_crop_fraction=1.,
                 channels=3,
                 mean=None):
        # The recommended batch size for this model
        self.batch_size = batch_size
        # The input size of this model
        self.input_size = input_size
        # A central crop fraction is expected by this model
        self.central_crop_fraction = central_crop_fraction
        # The number of channels in the input image expected by this model
        self.channels = channels
        # The mean to be subtracted from each image. By default, the per-channel ImageNet mean.
        # ImageNet mean value: np.array([124., 117., 104.]. Values are ordered RGB.
        self.mean = mean
        # The scalar to be multiplied from each image.
        self.scale = scale


class MVDescDataSpec(DataSpec):
    def __init__(self,
                 batch_size,
                 input_size,
                 scale=1.,
                 central_crop_fraction=1.,
                 channels=1,
                 mean=None,
                 view_num=3,
                 dim=128):
        super(MVDescDataSpec, self).__init__(batch_size=batch_size,
                                             input_size=input_size,
                                             scale=scale,
                                             central_crop_fraction=central_crop_fraction,
                                             channels=channels,
                                             mean=mean)
        self.view_num = view_num
        self.dim = dim

class ScoreNetDataSpec(DataSpec):
    def __init__(self,
                 batch_size,
                 image_size=(480, 640),
                 crop_size=(384, 512),
                 mean=128,
                 scale=0.00625,
                 channels=3,
                 focal_x=525.,
                 focal_y=525.,
                 u=320.,
                 v=240.):
        super(ScoreNetDataSpec, self).__init__(batch_size=batch_size,
                                               input_size=(0, 0),
                                               mean=mean,
                                               scale=scale,
                                               channels=channels)
        self.image_size = image_size
        self.crop_size = crop_size
        self.focal_x = focal_x
        self.focal_y = focal_y
        self.u = u
        self.v = v

def PoseNetSpec(batch_size):
    return DataSpec(batch_size=batch_size,
                    input_size=(224, 224),
                    channels=3,
                    mean=(124., 117., 104.))

def googlenet_spec():
    # set batch size based on gpu memory size
    nvmlInit()
    # TODO(tianwei): support multi-gpu
    handle = nvmlDeviceGetHandleByIndex(0)
    meminfo = nvmlDeviceGetMemoryInfo(handle)
    avail_mem = meminfo.free / 1024. ** 3
    bs = 8
    if avail_mem > 2:
        bs = 16
    if avail_mem > 4:
        bs = 32

    """Spec for GoogleNet."""
    return DataSpec(batch_size=bs,
                    input_size=(224, 224),
                    scale=1,
                    central_crop_fraction=1.0,
                    channels=3,
                    mean=[124., 117., 104.])


def l2net_spec():
    """Spec for L2Net."""
    return DataSpec(batch_size=8,
                    input_size=(64, 64),
                    scale=0.00625,
                    central_crop_fraction=0.5,
                    channels=1,
                    mean=128)

def mvdesc_spec():
    return MVDescDataSpec(batch_size=100,
                    input_size=(64, 64),
                    scale=0.00625,
                    central_crop_fraction=0.5,
                    channels=1,
                    mean=32,
                    view_num=3,
                    dim=32)

def tinyl2net_spec():
    """Spec for L2Net."""
    return DataSpec(batch_size=8,
                    input_size=(32, 32),
                    scale=0.00625,
                    central_crop_fraction=0.5,
                    channels=1,
                    mean=128)

def FlowNetS_spec(batch_size):
    """Spec for L2Net."""
    return DataSpec(batch_size=batch_size,
                    input_size=(384, 512),
                    scale=0.00625,
                    channels=3,
                    mean=128)


class SfMNetDataSpec(DataSpec):
    def __init__(self,
                 batch_size,
                 image_size=(480, 640),
                 crop_size=(384, 512),
                 focal_x=525.,
                 focal_y=525.,
                 u=320.,
                 v=240.):
        super(ScoreNetDataSpec, self).__init__(batch_size=batch_size, input_size=(0, 0))
        self.image_size = image_size
        self.crop_size = crop_size
        self.focal_x = focal_x
        self.focal_y = focal_y
        self.u = u
        self.v = v


# Collection of sample auto-generated models
# MODELS = (MatchNet)
MODELS = (
    GoogleNet,
    L2Net,
    MVDesc,
    ScoreNet,
    PoseNet,
    CoordFlowNet
)
# The corresponding data specifications for the sample models
# These specifications are based on how the models were trained.
MODEL_DATA_SPECS = {
    GoogleNet: googlenet_spec(),
    L2Net: l2net_spec(),
    MVDesc: mvdesc_spec(),
    ScoreNet: ScoreNetDataSpec(batch_size=24),
    PoseNet: PoseNetSpec(batch_size=1),
    FlowNetS: FlowNetS_spec(batch_size=8)
}


def get_models():
    """Returns a tuple of sample models."""
    return MODELS


def get_data_spec(model_instance=None, model_class=None):
    """Returns the data specifications for the given network."""
    model_class = model_class or model_instance.__class__
    return MODEL_DATA_SPECS[model_class]
