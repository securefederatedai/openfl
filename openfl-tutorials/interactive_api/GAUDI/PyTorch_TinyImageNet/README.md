# PyTorch TinyImageNet

## **Habana Tutorials**
Pytorch_tinyimagenet.ipynb placed under workspace folder contains the Habana specific adaptations.

 All the execution steps mention in last section (**III. How to run this tutorial**) remain same for HPU examples but as pre-requisite it needs some additional environment setup and Habana supported package installations which is explained below in **section I and II**.

**Note:** By default these experiments utilize only 1 HPU device
<br/>

 ## **I. Habana Setup**

 Please follow the instructions provided in the [Gaudi Installation Guide](https://docs.habana.ai/en/latest/Installation_Guide/index.html) to set up the environment including the ```$PYTHON``` environment variable.
<br/>
 
 **Note:** The example can be executed on any Gaudi configuration, DL1 on AWS or on an on-prem configuration. Currently, the validation has only been performed on DL1 instance on AWS created by following the instructions mentioned [here](https://docs.habana.ai/en/latest/AWS_EC2_DL1_and_PyTorch_Quick_Start/AWS_EC2_DL1_and_PyTorch_Quick_Start.html) . 

  **Test setup** - AWS DL1 instance, Habana 1.7, Ubuntu 20.04, Python 3.8 (hardcoded in [habanalabs-installer.sh](https://vault.habana.ai/ui/native/gaudi-installer/latest/habanalabs-installer.sh))
<br/>

 ## **II. HPU Adaptations For PyTorch Examples**

Please refer  [getting started with PyTorch](https://docs.habana.ai/en/latest/PyTorch/Getting_Started_with_PyTorch_and_Gaudi/Getting_Started_with_PyTorch.html) for detailed explaination of the Habana specific code modifications needed to run the model on HPU. Additionally the document provides sample PyTorch example with Habana specifc adaptations for reference, followed by set of instructions to execute it.
<br/>

### **Note**:

**There is a dependency in the order of execution (moving model to HPU and intializing optimizer). The workaround is to execute the below step before initializing any optimizers.**

```
model.to(device)
```
<br/>

## **III. How to run this tutorial (without TLS and locally as a simulation):**
<br/>

### 0. If you haven't done so already, install OpenFL in the virtual environment created during Habana setup, and upgrade pip:
  - For help with this step, visit the "Install the Package" section of the [OpenFL installation instructions](https://openfl.readthedocs.io/en/latest/install.html#install-the-package).
<br/>
 
### 1. Split terminal into 3 (1 terminal for the director, 1 for the envoy, and 1 for the experiment)
<br/> 

### 2. Do the following in each terminal:
   - Activate the virtual environment

   - If you are in a network environment with a proxy, ensure proxy environment variables are set in each of your terminals.
   - Navigate to the tutorial:
    
   ```sh
   cd openfl/openfl-tutorials/interactive_api/GAUDI/PyTorch_TinyImageNet
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
pip install -r requirements.txt
./start_envoy.sh env_one envoy_config_1.yaml
```

Optional: Run a second envoy in an additional terminal:
  - Ensure step 2 is complete for this terminal as well.
  - Run the second envoy:
```sh
cd envoy
./start_envoy.sh env_two envoy_config_2.yaml
```
<br/>

### 5. Now that your director and envoy terminals are set up, run the Jupyter Notebook in your experiment terminal:

```sh
cd workspace
jupyter lab pytorch_tinyimagenet.ipynb
```
- A Jupyter Server URL will appear in your terminal. In your browser, proceed to that link. Once the webpage loads, click on the pytorch_tinyimagenet.ipynb file. 
- To run the experiment, select the icon that looks like two triangles to "Restart Kernel and Run All Cells". 
- You will notice activity in your terminals as the experiment runs, and when the experiment is finished the director terminal will display a message that the experiment has finished successfully.  
