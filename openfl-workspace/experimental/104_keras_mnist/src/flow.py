# Copyright (C) 2020-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openfl.experimental.interface import FLSpec
from openfl.experimental.placement import aggregator, collaborator
from tensorflow.keras.layers import Flatten, Dense, Dropout, Conv2D, MaxPool2D
from tensorflow.keras.models import Sequential
import numpy as np


nb_classes = 10
model = Sequential([
    Conv2D(filters=32, kernel_size=(3, 3), activation="relu", input_shape=(28, 28, 1)),
    MaxPool2D(),
    Flatten(),
    Dense(512, activation="relu"),
    Dropout(0.2),
    Dense(512, activation="relu"),
    Dropout(0.2),
    Dense(nb_classes, activation="softmax"),
])
model.compile(optimizer="sgd", loss="categorical_crossentropy", metrics=["accuracy"])


def fedavg(models):
    new_model = models[0]
    state_dicts = [model.weights for model in models]
    state_dict = new_model.weights
    for idx, _ in enumerate(models[1].weights):
        state_dict[idx] = np.sum(np.array([state[idx]
                                 for state in state_dicts],
                                 dtype=object), axis=0) / len(models)
    new_model.set_weights(state_dict)
    return new_model


def inference(model, test_loader, batch_size):
    x_test, y_test = test_loader
    loss, accuracy = model.evaluate(
        x_test,
        y_test,
        batch_size=batch_size,
        verbose=0
    )
    accuracy_percentage = accuracy * 100
    print(f"Test set: Avg. loss: {loss}, Accuracy: {accuracy_percentage:.2f}%")
    return accuracy


class KerasMNISTFlow(FLSpec):
    def __init__(self, model, rounds=3, **kwargs):
        super().__init__(**kwargs)
        self.model = model
        self.n_rounds = rounds
        self.current_round = 1

    @aggregator
    def start(self):
        self.collaborators = self.runtime.collaborators
        self.next(self.aggregated_model_validation, foreach='collaborators')

    @collaborator
    def aggregated_model_validation(self):
        print(f'Performing aggregated model validation for collaborator {self.input}')
        self.agg_validation_score = inference(self.model, self.test_loader, self.batch_size)
        print(f'{self.input} value of {self.agg_validation_score}')
        self.next(self.train)

    @collaborator
    def train(self):
        x_train, y_train = self.train_loader
        history = self.model.fit(
            x_train, y_train,
            batch_size=self.batch_size,
            epochs=1,
            verbose=1,
        )
        self.loss = history.history["loss"][0]
        self.next(self.local_model_validation)

    @collaborator
    def local_model_validation(self):
        self.local_validation_score = inference(self.model, self.test_loader, self.batch_size)
        print(f'Doing local model validation for collaborator {self.input}:'
              + f' {self.local_validation_score}')
        self.next(self.join)

    @aggregator
    def join(self, inputs):
        self.average_loss = sum(input.loss for input in inputs) / len(inputs)
        self.aggregated_model_accuracy = sum(
            input.agg_validation_score for input in inputs) / len(inputs)
        self.local_model_accuracy = sum(
            input.local_validation_score for input in inputs) / len(inputs)
        print(f'Average aggregated model validation values = {self.aggregated_model_accuracy}')
        print(f'Average training loss = {self.average_loss}')
        print(f'Average local model validation values = {self.local_model_accuracy}')
        print("Taking FedAvg of models of all collaborators")
        self.model = fedavg([input.model for input in inputs])

        self.next(self.internal_loop)

    @aggregator
    def internal_loop(self):
        if self.current_round == self.n_rounds:
            self.next(self.end)
        else:
            self.current_round += 1
            self.next(self.aggregated_model_validation, foreach='collaborators')

    @aggregator
    def end(self):
        print('This is the end of the flow')
