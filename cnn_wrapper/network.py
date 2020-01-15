import numpy as np
import tensorflow as tf

# Zero padding in default. 'VALID' gives no padding.
DEFAULT_PADDING = 'SAME'


def layer(op):
    """Decorator for composable network layers."""

    def layer_decorated(self, *args, **kwargs):
        """Layer decoration."""
        # Automatically set a name if not provided.
        name = kwargs.setdefault('name', self.get_unique_name(op.__name__))
        # Figure out the layer inputs.
        if not self.terminals:
            raise RuntimeError('No input variables found for layer %s.' % name)
        elif len(self.terminals) == 1:
            layer_input = self.terminals[0]
        else:
            layer_input = list(self.terminals)
        # Perform the operation and get the output.
        layer_output = op(self, layer_input, *args, **kwargs)
        # Add to layer LUT.
        self.layers[name] = layer_output
        # This output is now the input for the next layer.
        self.feed(layer_output)
        # Return self for chained calls.
        return self

    return layer_decorated


class Network(object):
    """Class NetWork"""

    def __init__(self, inputs, is_training, dropout_rate=0.5, seed=None, reuse=False):
        # The input nodes for this network
        self.inputs = inputs
        # The current list of terminal nodes
        self.terminals = []
        # Mapping from layer names to layers
        self.layers = dict(inputs)
        # If true, the resulting variables are set as trainable
        self.trainable = is_training
        # If true, variables are shared between feature towers
        self.reuse = reuse
        # If true, layers like batch normalization or dropout are working in training mode
        self.training = is_training
        # Seed for randomness
        self.seed = seed
        # Dropout rate
        self.dropout_rate = dropout_rate
        self.setup()

    def setup(self):
        '''Construct the network. '''
        raise NotImplementedError('Must be implemented by the subclass.')

    def load(self, data_path, session, ignore_missing=False):
        '''Load network weights.
        data_path: The path to the numpy-serialized network weights
        session: The current TensorFlow session
        ignore_missing: If true, serialized weights for missing layers are ignored.
        '''
        data_dict = np.load(data_path).item()
        for op_name in data_dict:
            with tf.variable_scope(op_name, reuse=True):
                for param_name, data in data_dict[op_name].iteritems():
                    try:
                        var = tf.get_variable(param_name)
                        session.run(var.assign(data))
                    except ValueError:
                        if not ignore_missing:
                            raise

    def feed(self, *args):
        '''Set the input(s) for the next operation by replacing the terminal nodes.
        The arguments can be either layer names or the actual layers.
        '''
        assert args
        self.terminals = []
        for fed_layer in args:
            if isinstance(fed_layer, basestring):
                try:
                    fed_layer = self.layers[fed_layer]
                except KeyError:
                    raise KeyError('Unknown layer name fed: %s' % fed_layer)
            self.terminals.append(fed_layer)
        return self

    def get_output(self):
        '''Returns the current network output.'''
        return self.terminals[-1]

    def get_output_by_name(self, layer_name):
        '''
        Get graph node by layer name
        :param layer_name: layer name string
        :return: tf node
        '''
        return self.layers[layer_name]

    def get_unique_name(self, prefix):
        '''Returns an index-suffixed unique name for the given prefix.
        This is used for auto-generating layer names based on the type-prefix.
        '''
        ident = sum(t.startswith(prefix) for t, _ in self.layers.items()) + 1
        return '%s_%d' % (prefix, ident)

    def change_inputs(self, inputs):
        assert len(inputs) == 1
        for key in inputs:
            self.layers[key] = inputs[key]

    @layer
    def conv(self,
             input,
             kernel_size,
             filters,
             strides,
             name,
             relu=True,
             padding=DEFAULT_PADDING,
             biased=True):
        return tf.layers.conv2d(input,
                                filters=filters,
                                kernel_size=kernel_size,
                                strides=strides,
                                activation=tf.nn.relu if relu else None,
                                use_bias=biased,
                                padding=padding,
                                trainable=self.trainable,
                                reuse=self.reuse,
                                name=name)

    @layer
    def conv_bn(self,
                input,
                kernel_size,
                filters,
                strides,
                name,
                relu=True,
                center=False,
                padding=DEFAULT_PADDING):
        with tf.variable_scope(name):
            conv = tf.layers.conv2d(input,
                                    filters=filters,
                                    kernel_size=kernel_size,
                                    strides=strides,
                                    activation=None,
                                    use_bias=False,
                                    padding=padding,
                                    trainable=self.trainable,
                                    reuse=self.reuse,
                                    name=name)
            # note that offset is disabled in default
            # scale is typically unnecessary if next layer is relu.
            output = tf.layers.batch_normalization(conv,
                                               center=center,
                                               scale=False,
                                               training=self.training,
                                               fused=True,
                                               trainable=center,
                                               reuse=self.reuse)
            if relu:
                output = tf.nn.relu(output)
            return output

    @layer
    def conv3d(self,
               input,
               kernel_size,
               filters,
               strides,
               name,
               relu=True,
               padding=DEFAULT_PADDING,
               biased=True):
        return tf.layers.conv3d(input,
                                filters=filters,
                                kernel_size=kernel_size,
                                strides=strides,
                                activation=tf.nn.relu if relu else None,
                                use_bias=biased,
                                padding=padding,
                                trainable=self.trainable,
                                reuse=self.reuse,
                                name=name)

    @layer
    def conv3d_bn(self,
                  input,
                  kernel_size,
                  filters,
                  strides,
                  name,
                  relu=True,
                  center=False,
                  padding=DEFAULT_PADDING):
        with tf.variable_scope(name):
            conv = tf.layers.conv3d(input,
                                    filters=filters,
                                    kernel_size=kernel_size,
                                    strides=strides,
                                    activation=None,
                                    use_bias=False,
                                    padding=padding,
                                    trainable=self.trainable,
                                    reuse=self.reuse,
                                    name=name)
            bn = tf.layers.batch_normalization(conv,
                                               center=center,
                                               scale=False,
                                               training=self.training,
                                               fused=True,
                                               trainable=center,
                                               reuse=self.reuse)
            if relu:
                return tf.nn.relu(bn)
            else:
                return bn

    @layer
    def deconv3d(self,
                 input,
                 kernel_size,
                 filters,
                 strides,
                 name,
                 relu=True,
                 padding=DEFAULT_PADDING):
        return tf.layers.conv3d_transpose(input,
                                          filters=filters,
                                          kernel_size=kernel_size,
                                          strides=strides,
                                          activation=tf.nn.relu if relu else None,
                                          use_bias=False,
                                          padding=padding,
                                          trainable=self.trainable,
                                          reuse=self.reuse,
                                          name=name)

    @layer
    def deconv3d_bn(self,
                    input,
                    kernel_size,
                    filters,
                    strides,
                    name,
                    relu=True,
                    center=False,
                    padding=DEFAULT_PADDING):
        with tf.variable_scope(name):
            deconv = tf.layers.conv3d_transpose(input,
                                                filters=filters,
                                                kernel_size=kernel_size,
                                                strides=strides,
                                                activation=None,
                                                use_bias=False,
                                                padding=padding,
                                                trainable=self.trainable,
                                                reuse=self.reuse,
                                                name=name)
            bn = tf.layers.batch_normalization(deconv,
                                               center=center,
                                               scale=False,
                                               training=self.training,
                                               fused=True,
                                               trainable=center,
                                               reuse=self.reuse)
            if relu:
                return tf.nn.relu(bn)
            else:
                return bn

    @layer
    def relu(self, input, name):
        return tf.nn.relu(input, name=name)

    @layer
    def max_pool(self, input, pool_size, strides, name, padding=DEFAULT_PADDING):
        return tf.layers.max_pooling2d(input,
                                       pool_size=pool_size,
                                       strides=strides,
                                       padding=padding,
                                       name=name)

    @layer
    def avg_pool(self, input, pool_size, strides, name, padding=DEFAULT_PADDING):
        return tf.layers.average_pooling2d(input,
                                           pool_size=pool_size,
                                           strides=strides,
                                           padding=padding,
                                           name=name)

    @layer
    def l2_pool(self, input, pool_size, strides, name, padding=DEFAULT_PADDING):
        return tf.sqrt(tf.layers.average_pooling2d(
            tf.square(input),
            pool_size=pool_size,
            strides=strides,
            padding=padding,
            name=name))

    @layer
    def lrn(self, input, radius, alpha, beta, name, bias=1.0):
        return tf.nn.local_response_normalization(input,
                                                  depth_radius=radius,
                                                  alpha=alpha,
                                                  beta=beta,
                                                  bias=bias,
                                                  name=name)

    @layer
    def concat(self, inputs, axis, name):
        return tf.concat(values=inputs, axis=axis, name=name)

    @layer
    def add(self, inputs, name):
        return tf.add_n(inputs, name=name)

    @layer
    def fc(self, input, num_out, name, relu=True):
        return tf.layers.dense(input,
                               units=num_out,
                               activation=tf.nn.relu if relu else None,
                               trainable=self.trainable,
                               reuse=self.reuse,
                               name=name)

    @layer
    def softmax(self, input, name, dim=-1):
        input_shape = map(lambda v: v.value, input.get_shape())
        if len(input_shape) > 2:
            # For certain models (like NiN), the singleton spatial dimensions
            # need to be explicitly squeezed, since they're not broadcast-able
            # in TensorFlow's NHWC ordering (unlike Caffe's NCHW).
            if input_shape[1] == 1 and input_shape[2] == 1:
                input = tf.squeeze(input, squeeze_dims=[1, 2])
            else:
                raise ValueError('Rank 2 tensor input expected for softmax!')
        return tf.nn.softmax(input, dim=dim, name=name)

    @layer
    def batch_normalization(self, input, name, center=True, scale=True, relu=False):
        output = tf.layers.batch_normalization(input,
                                               center=center,
                                               scale=True,
                                               fused=True,
                                               trainable=self.trainable,
                                               reuse=self.reuse,
                                               name=name)
        if relu:
            output = tf.nn.relu(output)
        return output

    @layer
    def dropout(self, input, rate, name):
        return tf.layers.dropout(input,
                                 rate=rate,
                                 training=self.training,
                                 seed=self.seed,
                                 name=name)

    @layer
    def l2norm(self, input, name, dim=-1):
        return tf.nn.l2_normalize(input, dim=dim, name=name)

    @layer
    def squeeze(self, input, name=None):
        return tf.squeeze(input, name=name)

    @layer
    def maximum(self, inputs, name=None):
        max_val = inputs[0]
        for val in inputs[1:]:
            max_val = tf.maximum(max_val, val)
        return tf.maximum(max_val, inputs[0], name=name)

    @layer
    def tanh(self, input, name=None):
        return tf.tanh(input, name=name)

    @layer
    def reshape(self, input, shape, name=None):
        return tf.reshape(input, shape, name=name)

    @layer
    def slice(self, inputs, begin, size, name=None):
        return tf.slice(inputs, begin, size, name=name)

    @layer
    def add_n(self, inputs, name=None):
        return tf.add_n(inputs, name=name)

    @layer
    def conv_tanh(self,
             input,
             kernel_size,
             filters,
             strides,
             name,
             padding=DEFAULT_PADDING,
             biased=True):
        return tf.layers.conv2d(input,
                                filters=filters,
                                kernel_size=kernel_size,
                                strides=strides,
                                activation=tf.nn.tanh,
                                use_bias=biased,
                                padding=padding,
                                trainable=self.trainable,
                                reuse=self.reuse,
                                name=name)

    @layer
    def deconv(self,
             input,
             kernel_size,
             filters,
             strides,
             name,
             relu=True,
             padding=DEFAULT_PADDING,
             biased=True):
        return tf.layers.conv2d_transpose(input,
                                filters=filters,
                                kernel_size=kernel_size,
                                strides=strides,
                                activation=tf.nn.relu if relu else None,
                                use_bias=biased,
                                padding=padding,
                                trainable=self.trainable,
                                reuse=self.reuse,
                                name=name)
