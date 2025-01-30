# Pytorch CNN Histology Dataset Training with Fedcurv aggregation
The example code in this directory is used to train a Convolutional Neural Network using the Colorectal Histology dataset.
It uses the Pytorch framework and OpenFL's TaskTunner API.
The federation aggregates intermediate models using the [Fedcurv](https://arxiv.org/pdf/1910.07796)
aggregation algorithm, which performs well (Compared to [FedAvg](https://arxiv.org/abs/2104.11375)) when the datasets are not independent and identically distributed (IID) among collaborators.

Note that this example is similar to the one present in the `keras/histology` directory and is here to demonstrate the usage of a different aggregation algorithm using OpenFL's Taskrunner API.

The differenece between the two examples lies both in the `PyTorchCNNWithFedCurv` class which is used to define a stateful training method which uses an existing `FedCurv` object,
and in the `plan.yaml` file in which the training task is explicitly defined with a non-default aggregation method - `FedCurvWeightedAverage`.

## Running an example federation
The following instructions can be used to run the federation:
```
# Copy the workspace template, create collaborators and aggregator
fx workspace create --template torch/histology_fedcurv --prefix fedcurv
cd fedcurv fx workspace certify                                                      
fx aggregator generate-cert-request                                                           
fx aggregator certify --silent                                                   
fx plan initialize                                                                 

fx collaborator create -n collaborator1 -d 1
fx collaborator generate-cert-request -n collaborator1
fx collaborator certify -n collaborator1 --silent

fx collaborator create -n collaborator2 -d 2
fx collaborator generate-cert-request -n collaborator2
fx collaborator certify -n collaborator2 --silent

# Run aggregator and collaborators
fx aggregator start &
fx collaborator start -n collaborator1 &
fx collaborator start -n collaborator2
```