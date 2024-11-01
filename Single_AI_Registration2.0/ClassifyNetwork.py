import tensorflow as tf
from tensorflow.keras.layers import AlphaDropout, Activation, Dropout, Add, Conv2D, AveragePooling2D, Conv2DTranspose, Conv3DTranspose
from tensorflow.keras.layers import BatchNormalization, Activation, Add
from tensorflow.keras.regularizers import l2
import tensorflow.keras.backend as K
from tensorflow.keras import layers, initializers
import numpy as np
from tensorflow_train_v2.layers.initializers import he_initializer, selu_initializer
from tensorflow_train_v2.layers.layers import Sequential, UpSampling2DLinear, UpSampling2DCubic
from tensorflow_train_v2.networks.unet_base import UnetBase

def conv2d(layer_input, filters, conv_layers=2):
    d = Conv2D(filters, kernel_size=(3, 3), strides=(1, 1), padding='same')(layer_input)
    d = BatchNormalization()(d)
    d = Activation('relu')(d)

    for i in range(conv_layers - 1):
        d = Conv2D(filters, kernel_size=(3, 3), strides=(1, 1), padding='same')(d)
        d = BatchNormalization()(d)
        d = Activation('relu')(d)

    return d

def deconv2d(layer_input, filters):
    u = Conv2DTranspose(filters, 2, strides=(2, 2), padding='same')(layer_input)
    u = BatchNormalization()(u)
    u = Activation('relu')(u)
    return u


class SCNetLocal2D(UnetBase):
    """
    U-Net with average pooling and linear upsampling.
    """
    def __init__(self,
                 num_filters_base,
                 repeats=2,
                 dropout_ratio=0.0,
                 kernel_size=None,
                 activation=tf.nn.relu,
                 kernel_initializer=he_initializer,
                 alpha_dropout=False,
                 data_format='channels_first',
                 padding='same',
                 *args, **kwargs):
        super(SCNetLocal2D, self).__init__(*args, **kwargs)
        self.num_filters_base = num_filters_base
        self.repeats = repeats
        self.dropout_ratio = dropout_ratio
        self.kernel_size = kernel_size or [3] * 2
        self.activation = activation
        self.kernel_initializer = kernel_initializer
        self.alpha_dropout = alpha_dropout
        self.data_format = data_format
        self.padding = padding
        self.init_layers()

    def downsample(self, current_level):
        """
        Create and return downsample keras layer for the current level.
        :param current_level: The current level.
        :return: The keras.layers.Layer.
        """
        return AveragePooling2D([2] * 2, data_format=self.data_format)

    def upsample(self, current_level):
        """
        Create and return upsample keras layer for the current level.
        :param current_level: The current level.
        :return: The keras.layers.Layer.
        """
        return UpSampling2DLinear([2] * 2, data_format=self.data_format)

    def conv(self, current_level, postfix):
        """
        Create and return a convolution layer for the current level with the current postfix.
        :param current_level: The current level.
        :param postfix:
        :return:
        """
        return Conv2D(self.num_filters_base,
                      self.kernel_size,
                      name='conv' + postfix,
                      activation=self.activation,
                      data_format=self.data_format,
                      kernel_initializer=self.kernel_initializer,
                    #   kernel_regularizer=l2(l=1.0),
                      padding=self.padding)

    def combine(self, current_level):
        """
        Create and return combine keras layer for the current level.
        :param current_level: The current level.
        :return: The keras.layers.Layer.
        """
        # return ConcatChannels(data_format=self.data_format)
        return Add()

    def contracting_block(self, current_level):
        """
        Create and return the contracting block keras layer for the current level.
        :param current_level: The current level.
        :return: The keras.layers.Layer.
        """
        layers = []
        for i in range(self.repeats):
            layers.append(self.conv(current_level, str(i)))
            if self.alpha_dropout:
                layers.append(AlphaDropout(self.dropout_ratio))
            else:
                layers.append(Dropout(self.dropout_ratio))
        return Sequential(layers, name='contracting' + str(current_level))
    
    def parallel_block(self, current_level):
        layers = []
        layers.append(self.conv(current_level, str(0)))

        return Sequential(layers, name='parallel' + str(current_level))
                       
        

def activation_fn_output_kernel_initializer(activation):
    """
    Return actual activation function and kernel initializer.
    :param activation: Activation function string.
    :return: activation_fn, kernel_initializer
    """
    if activation == 'none':
        activation_fn = None
        kernel_initializer = tf.keras.initializers.TruncatedNormal(stddev=0.001)
    elif activation == 'tanh':
        activation_fn = tf.tanh
        kernel_initializer = tf.keras.initializers.TruncatedNormal(stddev=0.001)
    elif activation == 'abs_tanh':
        activation_fn = lambda x, *args, **kwargs: tf.abs(tf.tanh(x))
        kernel_initializer = tf.keras.initializers.TruncatedNormal(stddev=0.0001)
    elif activation == 'square_tanh':
        activation_fn = lambda x: tf.tanh(x * x)
        kernel_initializer = tf.keras.initializers.TruncatedNormal(stddev=0.05)
    elif activation == 'inv_gauss':
        activation_fn = lambda x: 1.0 - tf.math.exp(-tf.square(x))
        kernel_initializer = tf.keras.initializers.TruncatedNormal(stddev=0.05)
    elif activation == 'squash':
        a = 5
        b = 1
        l = 1
        activation_fn = lambda x: 1.0 / (l * b) * tf.math.log((1.0 + tf.math.exp(b * (x - (a - l / 2.0)))) / (1.0 + tf.math.exp(b * (x - (a + l / 2.0)))))
        kernel_initializer = tf.keras.initializers.TruncatedNormal(stddev=0.05)
    elif activation == 'sigmoid':
        activation_fn = lambda x: tf.nn.sigmoid(x - 5)
        kernel_initializer = tf.keras.initializers.TruncatedNormal(stddev=0.05)
    return activation_fn, kernel_initializer
    
class SCNet2D(tf.keras.Model):
    """
    The SpatialConfigurationNet.
    """
    def __init__(self,
                 num_labels,
                 num_filters_base=128,
                 num_levels=4,
                 activation='lrelu',
                 data_format='channels_first',
                 padding='same',
                 local_activation='none',
                 spatial_activation='none',
                 spatial_downsample=16,
                 dropout_ratio=0.5):
        """
        Initializer.
        :param num_labels: Number of outputs.
        :param num_filters_base: Number of filters for the local appearance and spatial configuration sub-networks.
        :param num_levels: Number of levels for the local appearance and spatial configuration sub-networks.
        :param activation: Activation of the convolution layers ('relu', 'lrelu', or 'selu').
        :param data_format: 'channels_first' or 'channels_last'
        :param padding: Convolution padding.
        :param local_activation: Activation function of local appearance output.
        :param spatial_activation: Activation function of spatial configuration output.
        :param spatial_downsample: Downsample factor for spatial configuration output.
        :param dropout_ratio: The dropout ratio after each convolution layer.
        """
        super(SCNet2D, self).__init__()
        self.unet = SCNetLocal2D
        self.data_format = data_format
        self.num_filters_base = num_filters_base
        if activation == 'relu':
            activation_fn = tf.nn.relu
            kernel_initializer = he_initializer
            alpha_dropout = False
        elif activation == 'lrelu':
            activation_fn = lambda x: tf.nn.leaky_relu(x, alpha=0.1)
            kernel_initializer = he_initializer
            alpha_dropout = False
        elif activation == 'selu':
            activation_fn = tf.nn.selu
            kernel_initializer = selu_initializer
            alpha_dropout = True
        local_activation_fn, local_heatmap_layer_kernel_initializer = activation_fn_output_kernel_initializer(local_activation)
        spatial_activation_fn, spatial_heatmap_layer_kernel_initializer = activation_fn_output_kernel_initializer(spatial_activation)
        self.downsampling_factor = spatial_downsample
        self.conv0 = Conv2D(num_filters_base, [3] * 2, name='conv0', kernel_initializer=kernel_initializer, activation=activation_fn, data_format=data_format, padding=padding)
        self.scnet_local = self.unet(num_filters_base=self.num_filters_base, num_levels=num_levels, kernel_initializer=kernel_initializer, 
                                     alpha_dropout=alpha_dropout, activation=activation_fn, dropout_ratio=dropout_ratio, data_format=data_format, padding=padding)
        self.local_heatmaps = Sequential([Conv2D(num_labels, [3] * 2, name='local_heatmaps', kernel_initializer=local_heatmap_layer_kernel_initializer, 
                                                 activation=None, data_format=data_format, padding=padding),
                                          Activation(local_activation_fn, dtype='float32', name='local_heatmaps')])
        self.downsampling = AveragePooling2D([self.downsampling_factor] * 2, name='local_downsampling', data_format=data_format)
        self.conv_spatial = Sequential([Conv2D(num_filters_base, [11] * 2, name='conv1', kernel_initializer=kernel_initializer, activation=activation_fn, data_format=data_format, padding=padding),
                                        Conv2D(num_filters_base, [11] * 2, name='conv2', kernel_initializer=kernel_initializer, activation=activation_fn, data_format=data_format, padding=padding), 
                                        Conv2D(num_filters_base, [11] * 2, name='conv3', kernel_initializer=kernel_initializer, activation=activation_fn, data_format=data_format, padding=padding)])
        self.spatial_heatmaps = Conv2D(num_labels, [11] * 2, name='spatial_heatmaps', kernel_initializer=spatial_heatmap_layer_kernel_initializer, 
                                       activation=None, data_format=data_format, padding=padding)
        self.upsampling = Sequential([UpSampling2DCubic([self.downsampling_factor] * 2, name='spatial_upsampling', data_format=data_format),
                                      Activation(spatial_activation_fn, dtype='float32', name='spatial_heatmaps')])

    #@tf.function
    def call(self, inputs, training=False, **kwargs):
        """
        Call model.
        :param inputs: Input tensors.
        :param training: If True, use training mode, otherwise testing mode.
        :param kwargs: Not used.
        :return: (heatmaps, local_heatmaps, spatial_heatmaps) tuple.
        """
        node = self.conv0(inputs, training=training)
        node = self.scnet_local(node, training=training)
        local_heatmaps = node = self.local_heatmaps(node, training=training)
        node = self.downsampling(node, training=training)
        node = self.conv_spatial(node, training=training)
        node = self.spatial_heatmaps(node, training=training)
        spatial_heatmaps = self.upsampling(node, training=training)
        heatmaps = local_heatmaps * spatial_heatmaps

        # return heatmaps, local_heatmaps, spatial_heatmaps
        # set the one output of the model
        return heatmaps