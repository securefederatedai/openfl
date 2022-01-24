# PyTorch Lightining tutorial for Generative Adverserial Network (GAN) Dataset

![GANs with MNIST](https://www.google.com/url?sa=i&url=https%3A%2F%2Fmafda.medium.com%2Fgans-conditional-gans-with-mnist-part-4-7f816d274d8c&psig=AOvVaw2_VaEsh2NJP2TUqfBngG5-&ust=1642876535302000&source=images&cd=vfe&ved=0CAsQjRxqFwoTCPjq2Jm-w_UCFQAAAAAdAAAAABAD "GANs with MNIST")


### 1. About model: Generative Adverserial Networks (GANs)
[Generative Adverserial Networks](https://arxiv.org/abs/1406.2661) or GANs were introduced to the machine learning community by Ian J. Goodfellow in 2014. The idea is to generate real-looking samples or images that resemble the training data. A GAN has three primary components: a generator model for generating new data, a discriminator model for classifying whether generated data are real faces, or fake, and the adversarial network that pits them against each other. The fundamental nature of these dual networks is to outplay each other. One generates images to fool the other while the other tries not to be fooled. Given enough time, the network becomes so good that it ends up making fake images as realistic as possible.

### 2. About framework: PyTorch Lightning
[Pytorch Lightning](https://www.pytorchlightning.ai/) is a framework built on top of PyTorch that allows the models to be scaled without the boilerplate.

### 3. About dataset: MNIST
[MNIST](http://yann.lecun.com/exdb/mnist/) database is a database of handwritten digits that has a training set of 60,000 examples, and a test set of 10,000 examples. It is a subset of a larger set available from NIST. The digits have been size-normalized and centered in a fixed-size image.

### 4. Using multiple optimizers
The example uses two different optimizers: one for discriminator and one for generator. The plugin to support multiple optimizers with OpenFL has been added. Note that in order to use PyTorch Lightning framework with a single optimizer, this plugin is not required.

### 5. Links
* [Original paper](https://arxiv.org/abs/1406.2661)
* [Original Github code](https://pytorch-lightning.readthedocs.io/en/stable/notebooks/lightning_examples/basic-gan.html)

### 6. How to run this tutorial (without TLS and locally as a simulation):

Go to example folder:
cd <openfl_folder>/openfl-tutorials/interactive_api/PyTorch_Lightning_MNIST_GAN

1. Run director:
```sh
cd director
bash start_director.sh
```

2. Run envoy:
```sh
cd envoy
bash start_envoy.sh
```

Optional: start second envoy:
 - Copy `envoy` to another place and run from there:
```sh
bash start_envoy_two.sh
```

3. Run `PyTorch_Lightning_GAN.ipynb` jupyter notebook:
```sh
cd workspace
jupyter notebook PyTorch_Lightning_GAN.ipynb
```
