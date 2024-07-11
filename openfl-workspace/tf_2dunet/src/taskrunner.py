# Copyright (C) 2020-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""You may copy this file as the starting point of your own model."""

import numpy as np
import tensorflow as tf

from openfl.utilities import Metric
from openfl.federated import TensorFlowTaskRunner

class UNet2D(TensorFlowTaskRunner):
    """Initialize.

    Args:
        **kwargs: Additional parameters to pass to the function

    """

    def __init__(self, initial_filters=16,
                 depth=5,
                 batch_norm=True,
                 use_upsampling=False,
                 **kwargs):
        """Initialize.

        Args:
            **kwargs: Additional parameters to pass to the function

        """
        super().__init__(**kwargs)

        self.model = self.create_model(
            input_shape=self.feature_shape,
            n_cl_out=self.data_loader.num_classes,
            initial_filters=initial_filters,
            use_upsampling=use_upsampling,
            depth=depth,
            batch_norm=batch_norm,
            **kwargs
        )
        self.initialize_tensorkeys_for_functions()

        self.model.summary(print_fn=self.logger.info, line_length=120)

    def create_model(self,
                     input_shape,
                     n_cl_out=1,
                     use_upsampling=False,
                     dropout=0.2,
                     print_summary=True,
                     seed=816,
                     depth=5,
                     dropout_at=(2, 3),
                     initial_filters=16,
                     batch_norm=True,
                     **kwargs):
        """Create the TensorFlow 3D U-Net CNN model.

        Args:
            input_shape (list): input shape of the data
            n_cl_out (int): Number of output classes in label (Default=1)
            **kwargs: Additional parameters to pass to the function

        """
        
        model = build_model(input_shape,
                            n_cl_out=n_cl_out,
                            use_upsampling=use_upsampling,
                            dropout=dropout,
                            print_summary=print_summary,
                            seed=seed,
                            depth=depth,
                            dropout_at=dropout_at,
                            initial_filters=initial_filters,
                            batch_norm=batch_norm)

        model.compile(
            loss=dice_loss,
            optimizer=tf.keras.optimizers.Adam(),
            metrics=[dice_coef, soft_dice_coef],
        )

        return model
    
    def train_(self, batch_generator, metrics: list = None, **kwargs):
        """Train single epoch.

        Override this function for custom training.

        Args:
            batch_generator: Generator of training batches.
                Each batch is a tuple of N train images and N train labels
                where N is the batch size of the DataLoader of the current TaskRunner instance.

            epochs: Number of epochs to train.
            metrics: Names of metrics to save.
        """
        history = self.model.fit(batch_generator,
                                 verbose=1,
                                 **kwargs)
        results = []
        for metric in metrics:
            value = np.mean([history.history[metric]])
            results.append(Metric(name=metric, value=np.array(value)))
        return results


def dice_coef(target, prediction, axis=(1, 2), smooth=0.0001):
    """
    Sorenson Dice.

    Returns
    -------
    dice coefficient (float)
    """
    prediction = tf.round(prediction)  # Round to 0 or 1

    intersection = tf.reduce_sum(target * prediction, axis=axis)
    union = tf.reduce_sum(target + prediction, axis=axis)
    numerator = tf.constant(2.) * intersection + smooth
    denominator = union + smooth
    coef = numerator / denominator

    return tf.reduce_mean(coef)


def soft_dice_coef(target, prediction, axis=(1, 2), smooth=0.0001):
    """
    Soft Sorenson Dice.

    Does not round the predictions to either 0 or 1.

    Returns
    -------
    soft dice coefficient (float)
    """
    intersection = tf.reduce_sum(target * prediction, axis=axis)
    union = tf.reduce_sum(target + prediction, axis=axis)
    numerator = tf.constant(2.) * intersection + smooth
    denominator = union + smooth
    coef = numerator / denominator

    return tf.reduce_mean(coef)


def dice_loss(target, prediction, axis=(1, 2), smooth=0.0001):
    """
    Sorenson (Soft) Dice loss.

    Using -log(Dice) as the loss since it is better behaved.
    Also, the log allows avoidance of the division which
    can help prevent underflow when the numbers are very small.

    Returns
    -------
    dice loss (float)
    """
    intersection = tf.reduce_sum(prediction * target, axis=axis)
    p = tf.reduce_sum(prediction, axis=axis)
    t = tf.reduce_sum(target, axis=axis)
    numerator = tf.reduce_mean(intersection + smooth)
    denominator = tf.reduce_mean(t + p + smooth)
    dice_loss = -tf.math.log(2. * numerator) + tf.math.log(denominator)

    return dice_loss


def build_model(input_shape,
                n_cl_out=1,
                use_upsampling=False,
                dropout=0.2,
                seed=816,
                depth=5,
                dropout_at=(2, 3),
                initial_filters=16,
                batch_norm=True,
                **kwargs):
    """Build the TensorFlow model.

    Args:
        input_tensor: input shape ot the model
        use_upsampling (bool): True = use bilinear interpolation;
                            False = use transposed convolution (Default=False)
        n_cl_out (int): Number of channels in output layer (Default=1)
        dropout (float): Dropout percentage (Default=0.2)
        print_summary (bool): True = print the model summary (Default = True)
        seed: random seed (Default=816)
        depth (int): Number of max pooling layers in encoder (Default=5)
        dropout_at: Layers to perform dropout after (Default=[2,3])
        initial_filters (int): Number of filters in first convolutional
        layer (Default=16)
        batch_norm (bool): True = use batch normalization (Default=True)
        **kwargs: Additional parameters to pass to the function
    """
    if (input_shape[0] % (2**depth)) > 0:
        raise ValueError(f'Crop dimension must be a multiple of 2^(depth of U-Net) = {2**depth}')

    inputs = tf.keras.layers.Input(input_shape, name='brats_mr_image')

    activation = tf.keras.activations.relu

    params = {'kernel_size': (3, 3), 'activation': activation,
              'padding': 'same',
              'kernel_initializer': tf.keras.initializers.he_uniform(seed=seed)}

    convb_layers = {}

    net = inputs
    filters = initial_filters
    for i in range(depth):
        name = f'conv{i + 1}a'
        net = tf.keras.layers.Conv2D(name=name, filters=filters, **params)(net)
        if i in dropout_at:
            net = tf.keras.layers.Dropout(dropout)(net)
        name = f'conv{i + 1}b'
        net = tf.keras.layers.Conv2D(name=name, filters=filters, **params)(net)
        if batch_norm:
            net = tf.keras.layers.BatchNormalization()(net)
        convb_layers[name] = net
        # only pool if not last level
        if i != depth - 1:
            name = f'pool{i + 1}'
            net = tf.keras.layers.MaxPooling2D(name=name, pool_size=(2, 2))(net)
            filters *= 2

    # do the up levels
    filters //= 2
    for i in range(depth - 1):
        if use_upsampling:
            up = tf.keras.layers.UpSampling2D(
                name=f'up{depth + i + 1}', size=(2, 2))(net)
        else:
            up = tf.keras.layers.Conv2DTranspose(name=f'transConv{depth + i + 1}',
                                                 filters=filters,
                                                 kernel_size=(2, 2),
                                                 strides=(2, 2),
                                                 padding='same')(net)
        net = tf.keras.layers.concatenate(
            [up, convb_layers[f'conv{depth - i - 1}b']],
            axis=-1
        )
        net = tf.keras.layers.Conv2D(
            name=f'conv{depth + i + 1}a',
            filters=filters, **params)(net)
        net = tf.keras.layers.Conv2D(
            name=f'conv{depth + i + 1}b',
            filters=filters, **params)(net)
        filters //= 2

    net = tf.keras.layers.Conv2D(name='prediction', filters=n_cl_out,
                                 kernel_size=(1, 1),
                                 activation='sigmoid')(net)

    model = tf.keras.models.Model(inputs=[inputs], outputs=[net])

    return model