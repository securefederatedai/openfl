import tensorflow as tf

import os

from unet3D import build_model, dice_coef, dice_loss, soft_dice_coef

from tf_brats_dataloader import DatasetGenerator

batch_size = 4
crop_dim = [64, 64, 64, 1]
data_path = "~/data/MICCAI_BraTS2020_TrainingData/"


brats_data = DatasetGenerator(crop_dim[:3],
                              data_path=os.path.abspath(os.path.expanduser(data_path)),
                              batch_size=batch_size,
                              train_test_split=0.8,
                              validate_test_split=0.5,
                              number_output_classes=1,
                              random_seed=816)

model = build_model(crop_dim)

model.compile(
                loss=dice_loss,
                optimizer = tf.keras.optimizers.Adam(learning_rate=0.01),
                metrics=[dice_coef, soft_dice_coef]
            )

model.fit(brats_data.ds_train, validation_data=brats.ds_val, epochs=10)