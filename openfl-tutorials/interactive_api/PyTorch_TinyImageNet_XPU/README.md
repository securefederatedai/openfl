# PyTorch_TinyImageNet

## **How to run this tutorial (without TLC and locally as a simulation):**
<br/>

Before we dive in, let's clarify some terms. XPU is a term coined by Intel to describe their line of computing devices, which includes CPUs, GPUs, FPGAs, and other accelerators. In this tutorial, we will be focusing on the Intel® Data Center GPU Max Series model, a GPU that is part of Intel's XPU lineup.

### 0a. If you haven't done so already, create a virtual environment, install OpenFL, and upgrade pip:
  - For help with this step, visit the "Install the Package" section of the [OpenFL installation instructions](https://openfl.readthedocs.io/en/latest/install.html#install-the-package).

<br/>

### 0b. Quick XPU Setup
  In this tutorial, when we refer to XPU, we are specifically referring to the Intel® Data Center GPU Max Series. When using the Intel® Extension for PyTorch* package, selecting the device as 'xpu' will refer to this Intel® Data Center GPU Max Series.
  
  For a successful setup, please follow the steps outlined in the [Installation Guide](https://intel.github.io/intel-extension-for-pytorch/xpu/2.1.10+xpu/tutorials/installation.html). This guide provides detailed information on system requirements and the installation process for the Intel® Extension for PyTorch. For a deeper understanding of features, APIs, and technical details, refer to the [Intel® Extension for PyTorch* Documentation](https://intel.github.io/intel-extension-for-pytorch/xpu/2.1.10+xpu/index.html).

Hardware Prerequisite: Intel® Data Center GPU Max Series.

This Jupyter Notebook has been tested and confirmed to work with the following versions:

  - intel-extension-for-pytorch==2.0.120 (xpu)
  - pytorch==2.0.1
  - torchvision==0.15.2

These versions were obtained from official Intel® channels.

Additionally, the XPU driver version used in testing was:

  - [XPU_Driver==803](https://dgpu-docs.intel.com/driver/installation.html)
  

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
   cd openfl/openfl-tutorials/interactive_api/PyTorch_TinyImageNet
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
./start_envoy.sh env_one envoy_config.yaml
```

Optional: Run a second envoy in an additional terminal:
  - Ensure step 2 is complete for this terminal as well.
  - Run the second envoy:
```sh
cd envoy
./start_envoy.sh env_two envoy_config.yaml
```

<br/>

### 5. Now that your director and envoy terminals are set up, run the Jupyter Notebook in your experiment terminal:

```sh
cd workspace
jupyter lab pytorch_tinyimagenet_XPU.ipynb
```
- A Jupyter Server URL will appear in your terminal. In your browser, proceed to that link. Once the webpage loads, click on the pytorch_tinyimagenet.ipynb file. 
- To run the experiment, select the icon that looks like two triangles to "Restart Kernel and Run All Cells". 
- You will notice activity in your terminals as the experiment runs, and when the experiment is finished the director terminal will display a message that the experiment has finished successfully.  
 
