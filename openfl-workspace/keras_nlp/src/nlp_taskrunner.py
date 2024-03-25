"""Copyright (C) 2020-2021 Intel Corporation
   SPDX-License-Identifier: Apache-2.0

Licensed subject to the terms of the separately executed evaluation
license agreement between Intel Corporation and you.
"""
from tensorflow import keras

from openfl.federated import KerasTaskRunner


def build_model(latent_dim, num_encoder_tokens, num_decoder_tokens, **kwargs):
    """
    Define the model architecture.

    Args:
        input_shape (numpy.ndarray): The shape of the data
        num_classes (int): The number of classes of the dataset
    Returns:
        tensorflow.python.keras.engine.sequential.Sequential: The model defined in Keras
    """
    encoder_inputs = keras.Input(shape=(None, num_encoder_tokens))
    encoder = keras.layers.LSTM(latent_dim, return_state=True)
    encoder_outputs, state_h, state_c = encoder(encoder_inputs)

    # We discard `encoder_outputs` and only keep the states.
    encoder_states = [state_h, state_c]

    # Set up the decoder, using `encoder_states` as initial state.
    decoder_inputs = keras.Input(shape=(None, num_decoder_tokens))

    # We set up our decoder to return full output sequences,
    # and to return internal states as well. We don't use the
    # return states in the training model, but we will use them in inference.
    decoder_lstm = keras.layers.LSTM(latent_dim, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
    decoder_dense = keras.layers.Dense(num_decoder_tokens, activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)

    # Define the model that will turn
    # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
    model = keras.Model([encoder_inputs, decoder_inputs], decoder_outputs)

    model.compile(
        optimizer=keras.optimizers.legacy.RMSprop(),
        loss='categorical_crossentropy', metrics=['accuracy']
    )

    return model


class KerasNLP(KerasTaskRunner):
    """A basic convolutional neural network model."""

    def __init__(self, latent_dim, **kwargs):
        """
        Init taskrunner.

        Args:
            **kwargs: Additional parameters to pass to the function
        """
        super().__init__(**kwargs)

        self.model = build_model(latent_dim,
                                 self.data_loader.num_encoder_tokens,
                                 self.data_loader.num_decoder_tokens,
                                 **kwargs)

        self.initialize_tensorkeys_for_functions()

        self.model.summary(print_fn=self.logger.info)

        self.logger.info(f'Train Set Size : {self.get_train_data_size()}')
