# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""You may copy this file as the starting point of your own model."""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import tensorflow as tf

from openfl.federated import KerasTaskRunner

from unet3D import build_model, dice_coef, dice_loss, soft_dice_coef

class TensorFlow3dUNet(KerasTaskRunner):
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

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
        self.dice_loss = dice_loss
        self.dice_coef = dice_coef
        self.soft_dice_coef = soft_dice_coef

        self.model = self.define_model(**kwargs)

        
        self.initialize_tensorkeys_for_functions()

    
    def define_model(self, **kwargs):

        
        self.model = build_model(self.feature_shape, **kwargs)

        self.model.compile(
                loss=self.dice_loss,
                optimizer=self.optimizer,
                metrics=[self.dice_coef, self.soft_dice_coef],
            )

        self.tvars = model.layers
        print(f'layer names: {[var.name for var in self.tvars]}')

        self.opt_vars = self.optimizer.variables()
        print(f'optimizer vars: {self.opt_vars}')

        # Two opt_vars for one tvar: gradient and square sum for RMSprop.
        self.fl_vars = self.tvars + self.opt_vars

        return model

if __name__ == "__main__":

    from tf_brats_dataloader import DatasetGenerator

    import argparse

    parser = argparse.ArgumentParser(
        description="Train 3D U-Net model", add_help=True,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--data_path",
                        default="~/data/MICCAI_BraTS2020_TrainingData/",  # Or wherever you unzipped the BraTS datset,
                        help="Root directory for BraTS 2020 dataset")
    parser.add_argument("--epochs",
                    type=int,
                    default=5,
                    help="Number of epochs")
    parser.add_argument("--crop_dim",
                    type=int,
                    default=64,
                    help="Crop all dimensions to this (height, width, depth)")
    parser.add_argument("--batch_size",
                    type=int,
                    default=4,
                    help="Training batch size")
    parser.add_argument("--train_test_split",
                    type=float,
                    default=0.80,
                    help="Train/test split (0-1)")
    parser.add_argument("--validate_test_split",
                    type=float,
                    default=0.50,
                    help="Validation/test split (0-1)")
    parser.add_argument("--number_input_channels",
                    type=int,
                    default=1,
                    help="Number of input channels")
    parser.add_argument("--number_output_classes",
                    type=int,
                    default=1,
                    help="Number of output classes/channels")
    parser.add_argument("--random_seed",
                    default=816,
                    help="Random seed for determinism")
    parser.add_argument("--print_model",
                    action="store_true",
                    default=True,
                    help="Print the summary of the model layers")
    parser.add_argument("--filters",
                        type=int,
                        default=16,
                        help="Number of filters in the first convolutional layer")
    parser.add_argument("--use_upsampling",
                        action="store_true",
                        default=False,
                        help="Use upsampling instead of transposed convolution")
    parser.add_argument("--use_batchnorm",
                        action="store_true",
                        default=True,
                        help="Use batch normalization")
    parser.add_argument("--saved_model_name",
                    default="saved_model_3DUnet",
                    help="Save model to this path")

    args = parser.parse_args()

    print(args)

    brats_data = DatasetGenerator([args.crop_dim, args.crop_dim, args.crop_dim],
                                data_path=os.path.abspath(os.path.expanduser(args.data_path)),
                                batch_size=args.batch_size,
                                train_test_split=args.train_test_split,
                                validate_test_split=args.validate_test_split,
                                number_input_channels=args.number_input_channels,
                                number_output_classes=args.number_output_classes,
                                random_seed=args.random_seed
                                )

    model = build_model([args.crop_dim, args.crop_dim, args.crop_dim, args.number_input_channels],
                        use_upsampling=args.use_upsampling,
                        n_cl_out=args.number_output_classes,
                        dropout=0.2,
                        print_summary=args.print_model,
                        seed=args.random_seed,
                        depth=5,
                        dropout_at=[2, 3],
                        initial_filters=args.filters,
                        batch_norm=args.use_batchnorm 
                        )

    model.compile(loss=dice_loss,
                  optimizer = tf.keras.optimizers.Adam(learning_rate=0.01),
                  metrics=[dice_coef, soft_dice_coef]
                 )

    checkpoint = tf.keras.callbacks.ModelCheckpoint(args.saved_model_name,
                                         verbose=1,
                                         save_best_only=True)

    # TensorBoard
    import datetime
    logs_dir = os.path.join("tensorboard_logs", 
                            datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    tb_logs = tf.keras.callbacks.TensorBoard(log_dir=logs_dir)

    callbacks = [checkpoint, tb_logs]            

    history = model.fit(brats_data.ds_train, 
                        validation_data=brats_data.ds_val, 
                        epochs=args.epochs,
                        callbacks=callbacks)