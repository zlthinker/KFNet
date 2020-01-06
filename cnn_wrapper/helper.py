from cnn_wrapper.SCoordNet import SCoordNet

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

class SCoordNetDataSpec(DataSpec):
    def __init__(self,
                 batch_size,
                 downsample = 8,
                 image_size=(480, 640),
                 crop_size=(480, 640),
                 mean=128,
                 scale=0.00625,
                 channels=3,
                 focal_x=525.,
                 focal_y=525.,
                 u=320.,
                 v=240.,
                 scene='fire'):
        super(SCoordNetDataSpec, self).__init__(batch_size=batch_size,
                                               input_size=(0, 0),
                                               mean=mean,
                                               scale=scale,
                                               channels=channels)
        self.downsample = downsample
        self.image_size = image_size
        self.crop_size = crop_size
        self.focal_x = focal_x
        self.focal_y = focal_y
        self.u = u
        self.v = v
        self.scene = scene

# Collection of sample auto-generated models
# MODELS = (MatchNet)
MODELS = (
    SCoordNet
)
# The corresponding data specifications for the sample models
# These specifications are based on how the models were trained.
MODEL_DATA_SPECS = {
    SCoordNet: SCoordNetDataSpec(batch_size=4)
}


def get_models():
    """Returns a tuple of sample models."""
    return MODELS


def get_data_spec(model_instance=None, model_class=None):
    """Returns the data specifications for the given network."""
    model_class = model_class or model_instance.__class__
    return MODEL_DATA_SPECS[model_class]
