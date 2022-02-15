# PyTorch Lightining tutorial for Generative Adverserial Network (GAN) Dataset

### 1. About model: Generative Adverserial Networks (GANs)

[Generative Adverserial Networks](https://arxiv.org/abs/1406.2661) or GANs were introduced to the
machine learning community by Ian J. Goodfellow in 2014. The idea is to generate real-looking
samples or images that resemble the training data. A GAN has three primary components: a Generator
model for generating new data from random data (noise), a discriminator model for classifying
whether generated data is real or fake, and the adversarial network that pits them against each
other. The fundamental nature of these dual networks is to outplay each other until the generator
starts generating real looking samples that the discriminator fails to differentiate.

### 2. About framework: PyTorch Lightning

[Pytorch Lightning](https://www.pytorchlightning.ai/) is a framework built on top of PyTorch that
allows the models to be scaled without the boilerplate.

### 3. About dataset: MNIST

[MNIST](http://yann.lecun.com/exdb/mnist/) database is a database of handwritten digits that has a
training set of 60,000 examples, and a test set of 10,000 examples. It is a subset of a larger set
available from NIST. The digits have been size-normalized and centered in a fixed-size image.

### 4. Using multiple optimizers

The example uses two different optimizers: one for discriminator and one for generator.
The [plugin](workspace/plugin_for_multiple_optimizers.py) to support multiple optimizers with
OpenFL has been added. Note that in order to use PyTorch Lightning framework with a single
optimizer, this plugin is NOT required.

### 5. Training Generator and Discriminator models separately

Cuurently, the tutorial shows how to train both the generator and the discriminator models
parallely. Individual models can be trained as well. To train only the generator, the flag '
train_gen_only' should be set to 1 and to train only the discriminator, 'train_disc_only' should be
set to 1.

### 5. Links

* [Original GAN paper](https://arxiv.org/abs/1406.2661)
* [Original PyTorch Lightning code](https://pytorch-lightning.readthedocs.io/en/stable/notebooks/lightning_examples/basic-gan.html)

### 6. How to run this tutorial (without TLS and locally as a simulation):

Go to example [folder](./)

```sh
export PYTORCH_LIGHTNING_MNIST_GAN=<openfl_folder>/openfl-tutorials/interactive_api/PyTorch_Lightning_MNIST_GAN
```

1. Run director:

```sh
cd $PYTORCH_LIGHTNING_MNIST_GAN/director
bash start_director.sh
```

2. Run envoy:

```sh
cd $PYTORCH_LIGHTNING_MNIST_GAN/envoy
pip install -r sd_requirements.txt
bash start_envoy.sh
```

Optional: start second envoy:

- Copy `$PYTORCH_LIGHTNING_MNIST_GAN/envoy` to another folder, change the config and envoy name in
  start_envoy.sh and run from there:

```sh
cd $PYTORCH_LIGHTNING_MNIST_GAN/envoy_two
bash start_envoy.sh
```

3. Run `PyTorch_Lightning_GAN.ipynb` jupyter notebook:

```sh
cd $PYTORCH_LIGHTNING_MNIST_GAN/workspace
jupyter lab PyTorch_Lightning_GAN.ipynb
```
