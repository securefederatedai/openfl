# Next Word Prediction Tutorial on Keras

### 0. GPU supporting

Currently GPU (with CUDA 11) is not supported by Tensorflow properly, so we disable CUDA in the
tutorial. Otherwise, you can use these charms in first
answer (https://stackoverflow.com/questions/41991101/importerror-libcudnn-when-running-a-tensorflow-program)
to fix your environment and enjoy GPU. Don't forget to
change `os.environ['CUDA_VISIBLE_DEVICES'] = '-1'` to a positive value.

As an option you can set CUDA variable for each envoy before starting
it: `export CUDA_VISIBLE_DEVICES=0`

### 1. Data

Different envoys could have different texts, there were used 3 books of fairy tales:

- Polish Fairy Tales by A. J. Gliński https://www.gutenberg.org/files/36668/36668-h/36668-h.htm
- English Fairy Tales by Joseph Jacobs https://www.gutenberg.org/cache/epub/7439/pg7439-images.html
- American Fairy Tales by L. FRANK BAUM https://www.gutenberg.org/files/4357/4357-h/4357-h.htm

### 2. Keras Model

At this point OpenFL maintains Sequential API and Functional API. Keras Submodel is not supported.
https://github.com/intel/openfl/issues/185

## To run experiment:

1. Create a folder for each envoy (they can be subfolders of `envoy` for simulation purposes or
   folders on different machines in a real-life setting), in our case we should create three
   folders.
2. Put a relevant `envoy_config` in each of the three folders and copy other files from `envoy`
   there as well.
3. Modify the `start_envoy` accordingly:
    1. change `env_one` to `env_two`, `env_three` (or any unique envoy names you like)
    2. `envoy_config_one.yaml` to  `envoy_config_two.yaml` and `envoy_config_three.yaml`.
4. Install requirements for each envoy: `pip install -r sd_requirements.txt`
5. Run the director: execute `start_director.sh` in `director`.
6. Run the envoys: execute `start_envoy.sh` in each envoy folder.
7. Run the notebook using JupyterLab in `workspace`.

```sh
cd workspace 
jupyter lab
```
