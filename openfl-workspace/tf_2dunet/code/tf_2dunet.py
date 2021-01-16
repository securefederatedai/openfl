# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""You may copy this file as the starting point of your own model."""

import tensorflow.compat.v1 as tf

from openfl.federated import TensorFlowTaskRunner


tf.disable_v2_behavior()


class TensorFlow2DUNet(TensorFlowTaskRunner):
    """Initialize.

    Args:
        **kwargs: Additional parameters to pass to the function

    """

    def __init__(self, **kwargs):
        """Initialize.

        Args:
            **kwargs: Additional parameters to pass to the function

        """
        super().__init__(**kwargs)

        self.create_model(**kwargs)
        self.initialize_tensorkeys_for_functions()

    def create_model(self,
                     training_smoothing=32.0,
                     validation_smoothing=1.0,
                     **kwargs):
        """Create the TensorFlow 2D U-Net model.

        Args:
            training_smoothing (float): (Default=32.0)
            validation_smoothing (float): (Default=1.0)
            **kwargs: Additional parameters to pass to the function

        """
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.intra_op_parallelism_threads = 112
        config.inter_op_parallelism_threads = 1
        self.sess = tf.Session(config=config)

        self.X = tf.placeholder(tf.float32, self.input_shape)
        self.y = tf.placeholder(tf.float32, self.input_shape)
        self.output = define_model(self.X, use_upsampling=True, **kwargs)

        self.loss = dice_coef_loss(self.y, self.output, smooth=training_smoothing)
        self.loss_name = dice_coef_loss.__name__
        self.validation_metric = dice_coef(
            self.y, self.output, smooth=validation_smoothing)
        self.validation_metric_name = dice_coef.__name__

        self.global_step = tf.train.get_or_create_global_step()

        self.tvars = tf.trainable_variables()

        # self.optimizer = tf.train.AdamOptimizer(4e-5, beta1=0.9, beta2=0.999)
        self.optimizer = tf.train.RMSPropOptimizer(1e-5)

        self.gvs = self.optimizer.compute_gradients(self.loss, self.tvars)
        self.train_step = self.optimizer.apply_gradients(self.gvs,
                                                         global_step=self.global_step)

        self.opt_vars = self.optimizer.variables()

        # FIXME: Do we really need to share the opt_vars?
        # Two opt_vars for one tvar: gradient and square sum for RMSprop.
        self.fl_vars = self.tvars + self.opt_vars

        self.initialize_globals()


def dice_coef(y_true, y_pred, smooth=1.0, **kwargs):
    """Dice coefficient.

    Calculate the Dice Coefficient

    Args:
        y_true: Ground truth annotation array
        y_pred: Prediction array from model
        smooth (float): Laplace smoothing factor (Default=1.0)
        **kwargs: Additional parameters to pass to the function

    Returns:
        float: Dice cofficient metric

    """
    intersection = tf.reduce_sum(y_true * y_pred, axis=[1, 2, 3])
    coef = (tf.constant(2.) * intersection + tf.constant(smooth)) / \
           (tf.reduce_sum(y_true, axis=[1, 2, 3]) + tf.reduce_sum(
               y_pred, axis=(1, 2, 3)) + tf.constant(smooth))
    return tf.reduce_mean(coef)


def dice_coef_loss(y_true, y_pred, smooth=1.0, **kwargs):
    """Dice coefficient loss.

    Calculate the -log(Dice Coefficient) loss

    Args:
        y_true: Ground truth annotation array
        y_pred: Prediction array from model
        smooth (float): Laplace smoothing factor (Default=1.0)
        **kwargs: Additional parameters to pass to the function

    Returns:
        float: -log(Dice cofficient) metric

    """
    intersection = tf.reduce_sum(y_true * y_pred, axis=(1, 2, 3))

    term1 = -tf.log(tf.constant(2.0) * intersection + smooth)
    term2 = tf.log(tf.reduce_sum(y_true, axis=(1, 2, 3))
                   + tf.reduce_sum(y_pred, axis=(1, 2, 3)) + smooth)

    term1 = tf.reduce_mean(term1)
    term2 = tf.reduce_mean(term2)

    loss = term1 + term2

    return loss


CHANNEL_LAST = True
if CHANNEL_LAST:
    concat_axis = -1
    data_format = 'channels_last'
else:
    concat_axis = 1
    data_format = 'channels_first'

tf.keras.backend.set_image_data_format(data_format)


def define_model(input_tensor,
                 use_upsampling=False,
                 n_cl_out=1,
                 dropout=0.2,
                 print_summary=True,
                 activation_function='relu',
                 seed=0xFEEDFACE,
                 depth=5,
                 dropout_at=[2, 3],
                 initial_filters=32,
                 batch_norm=True,
                 **kwargs):
    """Define the TensorFlow model.

    Args:
        input_tensor: input shape ot the model
        use_upsampling (bool): True = use bilinear interpolation;
                               False = use transposed convolution (Default=False)
        n_cl_out (int): Number of channels in input layer (Default=1)
        dropout (float): Dropout percentage (Default=0.2)
        print_summary (bool): True = print the model summary (Default = True)
        activation_function: The activation function to use after convolutional
         layers (Default='relu')
        seed: random seed (Default=0xFEEDFACE)
        depth (int): Number of max pooling layers in encoder (Default=5)
        dropout_at: Layers to perform dropout after (Default=[2,3])
        initial_filters (int): Number of filters in first convolutional
         layer (Default=32)
        batch_norm (bool): True = use batch normalization (Default=True)
        **kwargs: Additional parameters to pass to the function

    """
    # Set keras learning phase to train
    tf.keras.backend.set_learning_phase(True)

    # Don't initialize variables on the fly
    tf.keras.backend.manual_variable_initialization(False)

    inputs = tf.keras.layers.Input(tensor=input_tensor, name='Images')

    if activation_function == 'relu':
        activation = tf.nn.relu
    elif activation_function == 'leakyrelu':
        activation = tf.nn.leaky_relu

    params = dict(kernel_size=(3, 3), activation=activation,
                  padding='same', data_format=data_format,
                  kernel_initializer=tf.keras.initializers.he_uniform(seed=seed))

    convb_layers = {}

    net = inputs
    filters = initial_filters
    for i in range(depth):
        name = 'conv{}a'.format(i + 1)
        net = tf.keras.layers.Conv2D(name=name, filters=filters, **params)(net)
        if i in dropout_at:
            net = tf.keras.layers.Dropout(dropout)(net)
        name = 'conv{}b'.format(i + 1)
        net = tf.keras.layers.Conv2D(name=name, filters=filters, **params)(net)
        if batch_norm:
            net = tf.keras.layers.BatchNormalization()(net)
        convb_layers[name] = net
        # only pool if not last level
        if i != depth - 1:
            name = 'pool{}'.format(i + 1)
            net = tf.keras.layers.MaxPooling2D(name=name, pool_size=(2, 2))(net)
            filters *= 2

    # do the up levels
    filters //= 2
    for i in range(depth - 1):
        if use_upsampling:
            up = tf.keras.layers.UpSampling2D(
                name='up{}'.format(depth + i + 1), size=(2, 2))(net)
        else:
            up = tf.keras.layers.Conv2DTranspose(
                name='transConv6', filters=filters, data_format=data_format,
                kernel_size=(2, 2), strides=(2, 2), padding='same')(net)
        net = tf.keras.layers.concatenate(
            [up, convb_layers['conv{}b'.format(depth - i - 1)]],
            axis=concat_axis
        )
        net = tf.keras.layers.Conv2D(
            name='conv{}a'.format(depth + i + 1),
            filters=filters, **params)(net)
        net = tf.keras.layers.Conv2D(
            name='conv{}b'.format(depth + i + 1),
            filters=filters, **params)(net)
        filters //= 2

    net = tf.keras.layers.Conv2D(name='Mask', filters=n_cl_out,
                                 kernel_size=(1, 1), data_format=data_format,
                                 activation='sigmoid')(net)

    model = tf.keras.models.Model(inputs=[inputs], outputs=[net])

    if print_summary:
        print(model.summary())

    return net
