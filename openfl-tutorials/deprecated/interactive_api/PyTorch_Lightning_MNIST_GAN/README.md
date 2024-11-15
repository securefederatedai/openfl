# PyTorch Lightining tutorial for Generative Adverserial Network (GAN) Dataset

## **I. About model: Generative Adverserial Networks (GANs)**

[Generative Adverserial Networks](https://arxiv.org/abs/1406.2661) or GANs were introduced to the
machine learning community by Ian J. Goodfellow in 2014. The idea is to generate real-looking
samples or images that resemble the training data. A GAN has three primary components: a Generator
model for generating new data from random data (noise), a discriminator model for classifying
whether generated data is real or fake, and the adversarial network that pits them against each
other. The fundamental nature of these dual networks is to outplay each other until the generator
starts generating real looking samples that the discriminator fails to differentiate.

<br/>
<br/>

## **II. About framework: PyTorch Lightning**

[Pytorch Lightning](https://www.pytorchlightning.ai/) is a framework built on top of PyTorch that
allows the models to be scaled without the boilerplate.

<br/>
<br/>

## **III. About dataset: MNIST**

[MNIST](http://yann.lecun.com/exdb/mnist/) database is a database of handwritten digits that has a
training set of 60,000 examples, and a test set of 10,000 examples. It is a subset of a larger set
available from NIST. The digits have been size-normalized and centered in a fixed-size image.

<br/>
<br/>

## **IV. Using multiple optimizers**

The example uses two different optimizers: one for discriminator and one for generator.
The [plugin](workspace/plugin_for_multiple_optimizers.py) to support multiple optimizers with
OpenFL has been added. Note that in order to use PyTorch Lightning framework with a single
optimizer, this plugin is NOT required.

<br/>
<br/>

## **V. Training Generator and Discriminator models separately**

Cuurently, the tutorial shows how to train both the generator and the discriminator models
parallely. Individual models can be trained as well. To train only the generator, the flag '
train_gen_only' should be set to 1 and to train only the discriminator, 'train_disc_only' should be
set to 1.

<br/>
<br/>

## **VI. Links**

* [Original GAN paper](https://arxiv.org/abs/1406.2661)
* [Original PyTorch Lightning code](https://pytorch-lightning.readthedocs.io/en/stable/notebooks/lightning_examples/basic-gan.html)

<br/>
<br/>

## **VII. How to run this tutorial (without TLS and locally as a simulation):**

<br/>

### 0. If you haven't done so already, create a virtual environment, install OpenFL, and upgrade pip:
  - For help with this step, visit the "Install the Package" section of the [OpenFL installation instructions](https://openfl.readthedocs.io/en/latest/install.html#install-the-package).

<br/>
 
### 1. Split terminal into 3 (1 terminal for the director, 1 for the envoy, and 1 for the experiment)

<br/> 

### 2. Do the following in each terminal:
   - Activate the virtual environment from step 0:
   
   ```sh
   source venv/bin/activate
   ```
   - If you are in a network environment with a proxy, ensure proxy environment variables are set in each of your terminals.
   - Navigate to the tutorial:
    
   ```sh
   cd openfl/openfl-tutorials/interactive_api/PyTorch_Lightning_MNIST_GAN
   ```

<br/>

### 3. In the first terminal, run the director:

```sh
cd director
./start_director.sh
```

<br/>

### 4. In the second terminal, install requirements and run the envoy:

```sh
cd envoy
pip install -r sd_requirements.txt
```
 - If you have GPUs:
```sh
./start_envoy.sh env_one envoy_config.yaml
```
  - For no GPUs, use:
```sh
./start_envoy.sh env_one envoy_config_no_gpu.yaml
```


Optional: Run a second envoy in an additional terminal:
  - Ensure step 2 is complete for this terminal as well.
  - Repeat step 4 instructions above but change "env_one" name to "env_two" (or another name of your choice).

<br/>

### 5. Now that your director and envoy terminals are set up, run the Jupyter Notebook in your experiment terminal:

```sh
cd workspace
jupyter lab PyTorch_Lightning_GAN.ipynb
```
- A Jupyter Server URL will appear in your terminal. In your browser, proceed to that link. Once the webpage loads, click on the PyTorch_Lightning_GAN.ipynb file. 
- To run the experiment, select the icon that looks like two triangles to "Restart Kernel and Run All Cells". 
- You will notice activity in your terminals as the experiment runs, and when the experiment is finished the director terminal will display a message that the experiment has finished successfully.  
 
