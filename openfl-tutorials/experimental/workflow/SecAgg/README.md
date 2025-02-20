# Secure Aggregation - MNIST - FederatedRuntime

## **How to run this tutorial (without TLS and locally as a simulation):**
<br/>

**NOTE**: This is for demonstration purpose only. Use `LocalRuntime` to simulate the federation locally.
### 0. If you haven't done so already, create a virtual environment, install OpenFL, and upgrade pip:
  - For help with this step, visit the "Install the Package" section of the [OpenFL installation instructions](https://openfl.readthedocs.io/en/latest/get_started/installation.html).

<br/>
 
### 1. Split terminal into 4 (1 terminal for the director, 2 for the envoys, and 1 for the experiment)

<br/> 

### 2. Do the following in each terminal:
   - Activate the virtual environment from step 0:
   
   ```sh
   source venv/bin/activate
   ```
   - If you are in a network environment with a proxy, ensure proxy environment variables are set in each of your terminals.
   - Navigate to the tutorial:
    
   ```sh
   cd openfl/openfl-tutorials/experimental/workflow/SecAgg/
   ```

<br/>

### 3. In the first terminal, activate experimental features and run the director:

```sh
fx experimental activate
cd director
./start_director.sh
```

<br/>

### 4. In the second, and third terminals, run the envoys:

#### 4.1 Second terminal
```sh
cd Bengaluru
./start_envoy.sh Bengaluru Bengaluru_config.yaml
```

#### 4.2 Third terminal
```sh
cd Portland
./start_envoy.sh Portland Portland_config.yaml
```

<br/>

### 5. Now that your director and envoy terminals are set up, run the Jupyter Notebook in your experiment terminal:

```sh
cd workspace
jupyter lab MNIST_SecAgg.ipynb
```
- A Jupyter Server URL will appear in your terminal. In your browser, proceed to that link. Once the webpage loads, click on the MNIST_SecAgg.ipynb file. 
- To run the experiment, select the icon that looks like two triangles to "Restart Kernel and Run All Cells". 
- You will notice activity in your terminals as the experiment runs, and when the experiment is finished the director terminal will display a message that the experiment has finished successfully.  
 
