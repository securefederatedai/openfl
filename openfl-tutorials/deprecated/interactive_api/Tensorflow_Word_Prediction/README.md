# Next Word Prediction Tutorial on Keras

## I. GPU supporting

Currently GPU (with CUDA 11) is not supported by Tensorflow properly, so we disabled CUDA in the
tutorial. Otherwise, you can use these charms in the first answer of this Stack Overflow question (https://stackoverflow.com/questions/41991101/importerror-libcudnn-when-running-a-tensorflow-program)
to fix your environment and enjoy GPU. Don't forget to
change `os.environ['CUDA_VISIBLE_DEVICES'] = '-1'` to a positive value.

As an option you can set the CUDA variable for each envoy before starting
it: `export CUDA_VISIBLE_DEVICES=0`
<br/>
<br/>

## II. Data

Different envoys can have different texts, so in this tutorial each envoy uses one of these 3 fairy tale books:

- Polish Fairy Tales by A. J. Gliński https://www.gutenberg.org/files/36668/36668-h/36668-h.htm
- English Fairy Tales by Joseph Jacobs https://www.gutenberg.org/cache/epub/7439/pg7439-images.html
- American Fairy Tales by L. FRANK BAUM https://www.gutenberg.org/files/4357/4357-h/4357-h.htm
<br/>
<br/>

## III. Keras Model

At this point OpenFL maintains Sequential API and Functional API. Keras Submodel is not supported.
https://github.com/securefederatedai/openfl/issues/185
<br/>
<br/>

## IV. To run this experiment:
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
   cd openfl/openfl-tutorials/interactive_api/Tensorflow_Word_Prediction
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
./start_envoy.sh env_one envoy_config_one.yaml
```

Optional: Run a second or third envoy in additional terminals:
  - Ensure step 2 is complete for these terminals as well.
  - Follow step 4 for each envoy, changing the envoy name and config file accordingly. For example:
      - Envoy two would use: 
   ```sh
   ./start_envoy.sh env_two envoy_config_two.yaml
   ```
<br/>

### 5. Now that your director and envoy terminals are set up, run the Jupyter Notebook in your experiment terminal:

```sh
cd workspace
jupyter lab Tensorflow_Word_Prediction.ipynb
```
- A Jupyter Server URL will appear in your terminal. In your browser, proceed to that link. Once the webpage loads, click on the Tensorflow_Word_Prediction.ipynb file. 
- To run the experiment, select the icon that looks like two triangles to "Restart Kernel and Run All Cells". 
- You will notice activity in your terminals as the experiment runs, and when the experiment is finished the director terminal will display a message that the experiment has finished successfully.  
 
