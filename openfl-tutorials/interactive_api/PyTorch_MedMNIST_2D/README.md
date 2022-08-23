# MedMNIST 2D Classification Tutorial

![MedMNISTv2_overview](https://raw.githubusercontent.com/MedMNIST/MedMNIST/main/assets/medmnistv2.jpg)

For more details, please refer to the original paper:
**MedMNIST v2: A Large-Scale Lightweight Benchmark for 2D and 3D Biomedical Image Classification** ([arXiv](https://arxiv.org/abs/2110.14795))


### I. MedMNIST Installation and Requirements
Setup the required environments and install `medmnist` as a standard Python package from [PyPI](https://pypi.org/project/medmnist/):

    pip install medmnist

Or install from source:

    pip install --upgrade git+https://github.com/MedMNIST/MedMNIST.git

Check whether you have installed the latest [version](medmnist/info.py):

    >>> import medmnist
    >>> print(medmnist.__version__)

The code requires only common Python environments for machine learning. Basically, it was tested with
* Python 3 (>=3.6)
* PyTorch\==1.3.1
* numpy\==1.18.5, pandas\==0.25.3, scikit-learn\==0.22.2, Pillow\==8.0.1, fire, scikit-image

Higher (or lower) versions should also work (perhaps with minor modifications). 

### II. About model and experiments

We use a simple convolutional neural network and settings coming from [the experiments](https://github.com/MedMNIST/experiments) repository.

### III. How to run this tutorial (without TLC and locally as a simulation):
### 0. If you haven't done so already, create a virtual environment, install OpenFL, and upgrade pip:
  - For help with this step, visit the "Install the Package" section of the [OpenFL installation instructions](https://openfl.readthedocs.io/en/latest/install.html#install-the-package).
<br/>
 
### 1. Split terminal into 3 (1 terminal for the director, 1 for the envoy, and 1 for the experiment) 

### 2. Do the following in each terminal:
   - Activate the virtual environment from step 0:
   
   ```sh
   source venv/bin/activate
   ```
   - If you are in a network environment with a proxy, ensure proxy environment variables are set in each of your terminals.
   - Navigate to the tutorial:
    
   ```sh
   cd openfl/openfl-tutorials/interactive_api/PyTorch_MedMNIST_2D
   ```
<br/>

### 3. In the first terminal, run the director:

```sh
cd director
./start_director.sh
```
<br/>

### 4. In the second terminal, run the envoy:

```sh
cd envoy
./start_envoy.sh env_one envoy_config_one.yaml
```

Optional: Run a second envoy in an additional terminal:
  - Ensure step 2 is complete for this terminal as well.
  - Run the second envoy:
```sh
cd envoy
./start_envoy.sh env_two envoy_config_two.yaml
```
  - Notice that "env_one" was changed to "env_two", and "envoy_config_one.yaml" was changed to "envoy_config_two.yaml"
<br/>

### 5. In the third terminal (or forth terminal, if you chose to do two envoys) run the `Tensorflow_MNIST.ipynb` Jupyter Notebook:

```sh
cd workspace
jupyter lab PyTorch_MedMNIST_2D.ipynb
```
- A Jupyter Server URL will appear in your terminal. In your browser, proceed to that link. Once the webpage loads, click on the PyTorch_MedMNIST_2D.ipynb file. 
- To run the experiment, select the icon that looks like two triangles to "Restart Kernel and Run All Cells". 
- You will notice activity in your terminals as the experiments runs, and when the experiment is finished the director terminal will display a message that the experiment was finished successfully.  
 