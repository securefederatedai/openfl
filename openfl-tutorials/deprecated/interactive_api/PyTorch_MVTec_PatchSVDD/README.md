# Anomaly Detection with PatchSVDD for MVTec Dataset

![MVTec AD objects](https://www.mvtec.com/fileadmin/Redaktion/mvtec.com/company/research/datasets/dataset_overview_large.png "MVTec AD objects")

## **I. About the Dataset**

MVTec AD is a dataset for benchmarking anomaly detection methods with a focus on industrial
inspection. It contains over 5000 high-resolution images divided into fifteen different object and
texture categories. Each class contains 60 to 390 normal train images (defect free) and 40 to 167
test images (with various kinds of defects as well as images without defects). More info
at [MVTec dataset](https://www.mvtec.com/company/research/datasets/mvtec-ad). For each object, the
data is divided into 3 folders - 'train' (containing defect free training images), 'test'(
containing test images, both good and bad), 'ground_truth' (containing the masks of defected
images).

<br/>
<br/>

## **II. About the Model**

Two neural networks are used: an encoder and a classifier. The encoder is composed of convolutional
layers only. The classifier is a two layered MLP model having 128 hidden units per layer, and the
input to the classifier is a subtraction of the features of the two patches. The activation
function for both networks is a LeakyReLU with a α = 0.1. The encoder has a hierarchical structure.
The receptive field of the encoder is K = 64, and that of the embedded smaller encoder is K = 32.
Patch SVDD divides the images into patches with a size K and a stride S. The values for the strides
are S = 16 and S = 4 for the encoders with K = 64 and K = 32, respectively.

<br/>
<br/>

## **III. Links**

* [Original paper](https://arxiv.org/abs/2006.16067)
* [Original Github code](https://github.com/nuclearboy95/Anomaly-Detection-PatchSVDD-PyTorch/tree/934d6238e5e0ad511e2a0e7fc4f4899010e7d892)
* [MVTec ad dataset download link](https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420938113-1629952094/mvtec_anomaly_detection.tar.xz)

<br/>
<br/>

## **IV. How to run this tutorial (without TLS and locally as a simulation):**

<br/>

### **Note: An NVIDIA driver and GPUs are needed to run this tutorial unless configured otherwise.**

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
   cd openfl/openfl-tutorials/interactive_api/PyTorch_MVTec_PatchSVDD
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
./start_envoy.sh env_one envoy_config.yaml
```

Optional: Run a second envoy in an additional terminal:
  - Ensure step 2 is complete for this terminal as well.
  - Repeat step 4 instructions above but change "env_one" name to "env_two" (or another name of your choice).

<br/>

### 5. Now that your director and envoy terminals are set up, run the Jupyter Notebook in your experiment terminal:

```sh
cd workspace
jupyter lab PatchSVDD_with_Director.ipynb
```
- A Jupyter Server URL will appear in your terminal. In your browser, proceed to that link. Once the webpage loads, click on the PatchSVDD_with_Director.ipynb file. 
- To run the experiment, select the icon that looks like two triangles to "Restart Kernel and Run All Cells". 
- You will notice activity in your terminals as the experiment runs, and when the experiment is finished the director terminal will display a message that the experiment has finished successfully.  
 
