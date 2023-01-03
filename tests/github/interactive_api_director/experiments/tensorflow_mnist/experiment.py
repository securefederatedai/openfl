import time
import tensorflow as tf
# Create a federation
from openfl.interface.interactive_api.federation import Federation
from openfl.interface.interactive_api.experiment import TaskInterface, DataInterface, ModelInterface, FLExperiment
from tests.github.interactive_api_director.experiments.tensorflow_mnist.dataset import FedDataset
from tests.github.interactive_api_director.experiments.tensorflow_mnist.settings import model
from tests.github.interactive_api_director.experiments.tensorflow_mnist.settings import optimizer
from tests.github.interactive_api_director.experiments.tensorflow_mnist.settings import loss_fn
from tests.github.interactive_api_director.experiments.tensorflow_mnist.settings import train_acc_metric
from tests.github.interactive_api_director.experiments.tensorflow_mnist.settings import val_acc_metric
from tests.github.interactive_api_director.experiments.tensorflow_mnist.envoy.shard_descriptor import MNISTShardDescriptor
from copy import deepcopy


def run():
    # please use the same identificator that was used in signed certificate
    client_id = 'frontend'

    # 1) Run with API layer - Director mTLS 
    # If the user wants to enable mTLS their must provide CA root chain, and signed key pair to the federation interface
    # cert_chain = 'cert/root_ca.crt'
    # API_certificate = 'cert/frontend.crt'
    # API_private_key = 'cert/frontend.key'

    # federation = Federation(client_id='frontend', director_node_fqdn='localhost', director_port='50051',
    #                        cert_chain=cert_chain, api_cert=API_certificate, api_private_key=API_private_key)

    # --------------------------------------------------------------------------------------------------------------------

    # 2) Run with TLS disabled (trusted environment)
    # Federation can also determine local fqdn automatically
    federation = Federation(client_id=client_id, director_node_fqdn='localhost', director_port='50051', tls=False)

    shard_registry = federation.get_shard_registry()
    print(shard_registry)
    print(federation.target_shape)
    fed_dataset = FedDataset(train_bs=4, valid_bs=8)
    fed_dataset.shard_descriptor = MNISTShardDescriptor()
    for batch in fed_dataset.get_train_loader():
        samples, _ = batch
        for sample in samples:
            print(sample.shape)


    framework_adapter = 'openfl.plugins.frameworks_adapters.keras_adapter.FrameworkAdapterPlugin'
    MI = ModelInterface(model=model, optimizer=optimizer, framework_plugin=framework_adapter)


    def function_defined_in_notebook(some_parameter):
        print(f'Also I accept a parameter and it is {some_parameter}')


    TI = TaskInterface()
    # Task interface currently supports only standalone functions.
    @TI.register_fl_task(model='model', data_loader='train_dataset',
                        device='device', optimizer='optimizer')     
    def train(model, train_dataset, optimizer, device, loss_fn=loss_fn, warmup=False):

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


    @TI.register_fl_task(model='model', data_loader='val_dataset', device='device')     
    def validate(model, val_dataset, device):
        # Run a validation loop at the end of each epoch.
        for x_batch_val, y_batch_val in val_dataset:
            val_logits = model(x_batch_val, training=False)
            # Update val metrics
            val_acc_metric.update_state(y_batch_val, val_logits)
        val_acc = val_acc_metric.result()
        val_acc_metric.reset_states()
        print("Validation acc: %.4f" % (float(val_acc),))
                
        return {'validation_accuracy': val_acc,}
    # Save the initial model state
    train(model,fed_dataset.get_train_loader(), optimizer, 'cpu', warmup=True)
    initial_model = tf.keras.models.clone_model(model)



    # The Interactive API supports registering functions definied in main module or imported.


    # create an experimnet in federation
    experiment_name = 'mnist_test_experiment'
    fl_experiment = FLExperiment(federation=federation, experiment_name=experiment_name)
    # If I use autoreload I got a pickling error

    # The following command zips the workspace and python requirements to be transfered to collaborator nodes
    fl_experiment.start(model_provider=MI, 
                        task_keeper=TI,
                        data_loader=fed_dataset,
                        rounds_to_train=2,
                        opt_treatment='CONTINUE_GLOBAL')

    fl_experiment.stream_metrics()
    best_model = fl_experiment.get_best_model()
    fl_experiment.remove_experiment_data()
    validate(initial_model, fed_dataset.get_valid_loader(), 'cpu')
    validate(best_model, fed_dataset.get_valid_loader(), 'cpu')
