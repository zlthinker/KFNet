from cnn_wrapper.network import Network, layer
import tensorflow as tf

class SCoordNet(Network):
    def __init__(self, inputs, is_training, focal_x, focal_y, u, v, dropout_rate=0.5, seed=None, reuse=False):
        Network.__init__(self, inputs, is_training, dropout_rate, seed, reuse)
        self.focal_x = focal_x
        self.focal_y = focal_y
        self.u = u
        self.v = v

        images = inputs['input']
        shape = images.get_shape().as_list()
        self.batch_size = shape[0]
        self.height = shape[1]
        self.width = shape[2]

    def setup(self):
        (self.feed('input')
         .preprocess(name='preprocess')
         .conv(3, 64, 1, name='conv1a')
         .conv(3, 64, 1, name='conv1b')
         .conv(3, 256, 2, name='conv2a')
         .conv(3, 256, 1, name='conv2b')
         .conv(3, 512, 2, name='conv3a')
         .conv(3, 512, 1, name='conv3b')
         .conv(3, 1024, 2, name='conv4a')
         .conv(3, 1024, 1, name='conv4b')
         .conv(3, 512, 1, name='conv5')
         .conv(3, 256, 1, name='conv6')
         .conv(1, 128, 1, name='conv7')
         .conv(1, 4, 1, relu=False, name='prediction'))

    @layer
    def preprocess(self, input, name):
        input = tf.multiply(tf.subtract(input, 128.0), 0.00625, name=name)
        return input

    def GetOutput(self):
        prediction = self.get_output_by_name('prediction')
        coord_map = tf.slice(prediction, [0, 0, 0, 0], [-1, -1, -1, 3], name='coord')
        uncertainty_map = tf.slice(prediction, [0, 0, 0, 3], [-1, -1, -1, 1], name='uncertainty')
        uncertainty_map = tf.exp(uncertainty_map)
        return coord_map, uncertainty_map








