# MNIST Classification Tutorial

![mnist digits](http://i.ytimg.com/vi/0QI3xgXuB-Q/hqdefault.jpg "MNIST Digits")

## I. About the dataset

It is a dataset of 60,000 small square 28×28 pixel grayscale images of handwritten single digits
between 0 and 9. More info at [wiki](https://en.wikipedia.org/wiki/MNIST_database).

## II. About the model

We use a simple fully-connected neural network defined at
[layers.py](./workspace/layers.py) file.

## III. How to run this tutorial (without TLC and locally as a simulation):
<br/>

### 0. If you haven't done so already, create a virtual environment, install OpenFL, and upgrade pip:
  - For help with this step, visit the "Install the Package" section of the [OpenFL installation instructions](https://openfl.readthedocs.io/en/latest/install.html#install-the-package).
<br/>
 
### 1. Split terminal into 3 (1 terminal for the director, 1 for the envoy, and 1 for the experiment) and do the following in each terminal:
   - Activate the virtual environment from step 0:
   
   ```sh
   source venv/bin/activate
   ```
   - If you are in a network environment with a proxy, ensure proxy environment variables are set in each of your terminals.
   - Navigate to the tutorial:
    
   ```sh
   cd openfl/openfl-tutorials/interactive_api/Tensorflow_MNIST
   ```

    
<br/>

### 2. In the first terminal, run the director:

```sh
cd director
./start_director.sh
```
<br/>

### 3. In the second terminal, run the envoy:

```sh
cd envoy
./start_envoy.sh env_one envoy_config_one.yaml
```

Optional: Run a second envoy in an additional terminal:
  - Ensure steps 0 and 1 are complete for this terminal as well.
  - Run the second envoy:
```sh
cd envoy
./start_envoy.sh env_two envoy_config_two.yaml
```
  - Notice that "env_one" was changed to "env_two", and "envoy_config_one.yaml" was changed to "envoy_config_two.yaml"
<br/>

### 4. In the third terminal (or forth terminal, if you chose to do two envoys) run the `Tensorflow_MNIST.ipynb` Jupyter Notebook:

```sh
cd workspace
jupyter lab Tensorflow_MNIST.ipynb
```
