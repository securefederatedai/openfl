# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Register tasks."""

from openfl.interface.interactive_api.experiment import TaskInterface
from tests.github.interactive_api.experiments.tensorflow_mnist.settings import loss_fn, \
    train_acc_metric, val_acc_metric

task_interface = TaskInterface()


@task_interface.register_fl_task(model='model', data_loader='train_dataset',
                                 device='device', optimizer='optimizer')
def train(model, train_dataset, optimizer, device, loss_fn=loss_fn, warmup=False):
    """Train."""
    import tensorflow as tf

    # Iterate over the batches of the dataset.
    for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
        with tf.GradientTape() as tape:
            logits = model(x_batch_train, training=True)
            loss_value = loss_fn(y_batch_train, logits)
        grads = tape.gradient(loss_value, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))

        # Update training metric.
        train_acc_metric.update_state(y_batch_train, logits)

        # Log every 200 batches.
        if step % 200 == 0:
            print(
                "Training loss (for one batch) at step %d: %.4f"
                % (step, float(loss_value))
            )
            print("Seen so far: %d samples" % ((step + 1) * 64))
        if warmup:
            break

    # Display metrics at the end of each epoch.
    train_acc = train_acc_metric.result()
    print("Training acc over epoch: %.4f" % (float(train_acc),))

    # Reset training metrics at the end of each epoch
    train_acc_metric.reset_states()

    return {'train_acc': train_acc}


@task_interface.register_fl_task(model='model', data_loader='val_dataset', device='device')
def validate(model, val_dataset, device):
    """Validate."""
    # Run a validation loop at the end of each epoch.
    for x_batch_val, y_batch_val in val_dataset:
        val_logits = model(x_batch_val, training=False)
        # Update val metrics
        val_acc_metric.update_state(y_batch_val, val_logits)
    val_acc = val_acc_metric.result()
    val_acc_metric.reset_states()
    print("Validation acc: %.4f" % (float(val_acc),))

    return {'validation_accuracy': val_acc}
