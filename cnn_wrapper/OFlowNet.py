from cnn_wrapper.network import Network
import tensorflow as tf

# Input BxHxWxN cost volume
# Outout BxHxWx3 coordinate map & BxHxWx1 uncertainty map
class OFlowNet(Network):
    def __init__(self, inputs, window_area, is_training, reuse=False):
        images = inputs['input']
        shape = images.get_shape().as_list()
        self.batch_size = shape[0]
        self.height = shape[1]
        self.width = shape[2]
        self.window_area = window_area

        Network.__init__(self, inputs, is_training, reuse=reuse)

    def setup(self):
        (self.feed('input')
         .conv(3, 32, 1, name='conv0')  # 8x8x16
         .conv(3, 32, 2, name='conv1a') # 4x4x32
         .conv(3, 32, 1, name='conv1b') # 4x4x32
         .conv(3, 64, 2, name='conv2a') # 2x2x64
         .conv(3, 64, 1, name='conv2b') # 2x2x64
         .conv(3, 128, 2, name='conv3a')  # 1x1x128
         .conv(3, 128, 1, name='conv3b')  # 1x1x128
         .deconv(3, 64, 2, name='upconv2'))    # 2x2x64

        (self.feed('upconv2', 'conv2b')
         .concat(-1, name='concat2')    # 2x2x128
         .conv(3, 64, 1, name='conv4')  # 2x2x64
         .deconv(3, 32, 2, name='upconv1')) # 4x4x32

        (self.feed('upconv1', 'conv1b')
         .concat(-1, name='concat1')    # 4x4x64
         .conv(3, 32, 1, name='conv5')  # 4x4x32
         .deconv(3, 16, 2, name='upconv0'))  # 8x8x16

        (self.feed('upconv0', 'conv0')
         .concat(-1, name='concat0')    # 8x8x32
         .conv(3, 16, 1, name='conv6')   # 8x8x16
         .conv(3, 1, 1, relu=False, name='prediction')) # 8x8x2

    def GetOutput(self):
        output = self.get_output_by_name('prediction')  # BHWx8x8x2
        prob_map = tf.slice(output, [0, 0, 0, 0], [-1, -1, -1, 1])  # BHWx8x8x1
        prob_map = tf.reshape(prob_map, shape=[self.batch_size, self.window_area])
        prob_map = tf.nn.softmax(prob_map, axis=-1, name='prob')  # BHWx(window_area)
        feat = self.get_output_by_name('conv3b')  # BHWx1x1x128
        feat = tf.reshape(feat, [self.batch_size, -1])  # BHWx128
        fc1 = tf.layers.dense(feat, 64, activation=tf.nn.relu, trainable=self.trainable, reuse=self.reuse,
                                       name='fc1')
        fc2 = tf.layers.dense(fc1, 32, activation=tf.nn.relu, trainable=self.trainable, reuse=self.reuse,
                                       name='fc2')
        uncertainty = tf.layers.dense(fc2, 1, activation=None, trainable=self.trainable, reuse=self.reuse,
                                               name='uncertainty')  # BHWx1
        uncertainty = tf.exp(uncertainty) * 1e-2
        return prob_map, uncertainty