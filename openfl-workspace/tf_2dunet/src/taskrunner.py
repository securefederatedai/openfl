# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""You may copy this file as the starting point of your own model."""

import tensorflow as tf
import keras

from openfl.federated import KerasTaskRunner


class TensorFlow2DUNet(KerasTaskRunner):
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
        self.input_shape = self.data_loader.get_feature_shape()
        self.model = self.define_model(self.input_shape, use_upsampling=True, **kwargs)
        self.model.summary(print_fn=self.logger.info, line_length=120)

    def define_model(self, input_shape,
                use_upsampling=False,
                n_cl_out=1,
                dropout=0.2,
                print_summary=True,
                activation_function='relu',
                seed=0xFEEDFACE,
                depth=5,
                dropout_at=None,
                initial_filters=32,
                batch_norm=True,
                **kwargs):
        """Define the TensorFlow model.

        Args:
            input_shape: input shape of the model
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
        if dropout_at is None:
            dropout_at = [2, 3]

        inputs = keras.layers.Input(shape=input_shape, name='Images')

        if activation_function == 'relu':
            activation = tf.nn.relu
        elif activation_function == 'leakyrelu':
            activation = tf.nn.leaky_relu

        params = {
            'activation': activation,
            'data_format': data_format,
            'kernel_initializer': keras.initializers.he_uniform(seed=seed),
            'kernel_size': (3, 3),
            'padding': 'same',
        }

        convb_layers = {}

        net = inputs
        filters = initial_filters
        for i in range(depth):
            name = f'conv{i + 1}a'
            net = keras.layers.Conv2D(name=name, filters=filters, **params)(net)
            if i in dropout_at:
                net = keras.layers.Dropout(dropout)(net)
            name = f'conv{i + 1}b'
            net = keras.layers.Conv2D(name=name, filters=filters, **params)(net)
            if batch_norm:
                net = keras.layers.BatchNormalization()(net)
            convb_layers[name] = net
            # only pool if not last level
            if i != depth - 1:
                name = f'pool{i + 1}'
                net = keras.layers.MaxPooling2D(name=name, pool_size=(2, 2))(net)
                filters *= 2

        # do the up levels
        filters //= 2
        for i in range(depth - 1):
            if use_upsampling:
                up = keras.layers.UpSampling2D(
                    name=f'up{depth + i + 1}', size=(2, 2))(net)
            else:
                up = keras.layers.Conv2DTranspose(
                    name='transConv6', filters=filters, data_format=data_format,
                    kernel_size=(2, 2), strides=(2, 2), padding='same')(net)
            net = keras.layers.concatenate(
                [up, convb_layers[f'conv{depth - i - 1}b']],
                axis=concat_axis
            )
            net = keras.layers.Conv2D(
                name=f'conv{depth + i + 1}a',
                filters=filters, **params)(net)
            net = keras.layers.Conv2D(
                name=f'conv{depth + i + 1}b',
                filters=filters, **params)(net)
            filters //= 2
        net = keras.layers.Conv2D(name='Mask', filters=n_cl_out,
                                    kernel_size=(1, 1), data_format=data_format,
                                    activation='sigmoid')(net)
        model = keras.models.Model(inputs=[inputs], outputs=[net])


        self.optimizer = keras.optimizers.RMSprop(1e-2)
        model.compile(
            loss=self.dice_coef_loss,
            optimizer=self.optimizer,
            metrics=["acc"],
        )

        return model

    def dice_coef_loss(self, y_true, y_pred):
        """Dice coefficient loss.

        Calculate the -log(Dice Coefficient) loss

        Args:
            y_true: Ground truth annotation array
            y_pred: Prediction array from model
            smooth (float): Laplace smoothing factor (Default=1.0)
        Returns:
            float: -log(Dice cofficient) metric
        """
        intersection = tf.reduce_sum(y_true * y_pred, axis=(1, 2, 3))
        smooth=1.0
        term1 = -tf.math.log(tf.constant(2.0) * intersection + smooth)
        term2 = tf.math.log(tf.reduce_sum(y_true, axis=(1, 2, 3))
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

