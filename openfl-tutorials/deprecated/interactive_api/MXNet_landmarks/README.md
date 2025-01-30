# MXNet Facial Keypoints Detection tutorial
---
**Note:**

Please pay attention that this task uses the dataset from Kaggle. To get the dataset you
will need a Kaggle account and accept "Facial Keypoints Detection" [competition rules](https://www.kaggle.com/c/facial-keypoints-detection/rules).

---

This tutorial shows how to use any other framework, different from already supported PyTorch and TensorFlow, together with OpenFl.

## Installation of Kaggle API credentials

**Before the start please make sure that you installed sd_requirements.txt on your virtual
environment on an envoy machine.**

To use the [Kaggle API](https://github.com/Kaggle/kaggle-api), sign up for
a [Kaggle account](https://www.kaggle.com). Then go to the `'Account'` tab of your user
profile `(https://www.kaggle.com/<username>/account)` and select `'Create API Token'`. This will
trigger the download of `kaggle.json`, a file containing your API credentials. Place this file in
the location `~/.kaggle/kaggle.json`

For your security, ensure that other users of your computer do not have read access to your
credentials. On Unix-based systems you can do this with the following command:

`chmod 600 ~/.kaggle/kaggle.json`

If you need proxy add "proxy": `"http://<ip_addr:port>" in kaggle.json`. It should looks like
that: `{"username":"your_username","key":"token", "proxy": "ip_addr:port"}`

*Information about Kaggle API settings has been taken from kagge-api [readme](https://github.com/Kaggle/kaggle-api).*

*Useful [link](https://github.com/Kaggle/kaggle-api/issues/6) for a problem with proxy settings.*

### 1. About dataset

All information about the dataset you may find
on [link](https://www.kaggle.com/c/facial-keypoints-detection/data)

### 2. Adding support for a third-party framework

You need to write your own adapter class which is based on `FrameworkAdapterPluginInterface` [class](https://github.com/securefederatedai/openfl/blob/develop/openfl/plugins/frameworks_adapters/framework_adapter_interface.py). This class should contain at least two methods:

 - `get_tensor_dict(model, optimizer=None)` - extracts tensor dict from a model and optionally[^1] an optimizer. The resulting tensors must be converted to **dict{str: numpy.array}** for forwarding and aggregation.

  - `set_tensor_dict(model, tensor_dict, optimizer=None, device=None)` - sets aggregated numpy arrays into the model or model and optimizer. To do so it gets `tensor_dict` variable as **dict{str: numpy.array}** and should convert it into suitable for your model or model and optimizer tensors. After that, it must load the prepared parameters into the model/model and optimizer. 

 Your adapter should be placed in workspace directory. When you create `ModelInterface` class object at the `'***.ipunb'`, place the name of your adapter to the input parameter `framework_plugin`. Example: 
 ```py
 framework_adapter = 'mxnet_adapter.FrameworkAdapterPlugin'

 MI = ModelInterface(model=model, optimizer=optimizer,
                    framework_plugin=framework_adapter)
```

[^1]: Whether or not to forward the optimizer parameters is set in the `start` method (FLExperiment [class](https://github.com/securefederatedai/openfl/blob/develop/openfl/interface/interactive_api/experiment.py) object, parameter `opt_treatment`).

### Run experiment

1. Create a folder for each `envoy`.
2. Put a relevant envoy_config in each of the n folders (n - number of envoys which you would like
   to use, in this tutorial there is two of them, but you may use any number of envoys) and copy
   other files from `envoy` folder there as well.
3. Modify each `envoy` accordingly:

    - At `start_envoy.sh` change env_one to env_two (or any unique `envoy` names you like)

    - Put a relevant envoy_config `envoy_config_one.yaml` or `envoy_config_two.yaml` (or any other
      config file name consistent to the configuration file that is called in `start_envoy.sh`).
4. Make sure that you installed requirements for each `envoy` in your virtual
   environment: `pip install -r sd_requirements.txt`
5. Run the `director`: 
    ```sh
    cd director_folder
    ./start_director.sh
    ```

6. Run the `envoys`: 
    ```sh
    cd envoy_folder
    ./start_envoy.sh env_one shard_config_one.yaml
    ```
    If kaggle-API setting are
    correct the download of the dataset will be started. If this is not the first `envoy` launch
    then the dataset will be redownloaded only if some part of the data are missing.

7. Run the [MXNet_landmarks.ipynb](workspace/MXNet_landmarks.ipynb) notebook using
   Jupyter lab in a prepared virtual environment. For more information about preparation virtual
   environment look **[
   Preparation virtual environment](#preparation-virtual-environment)**
   .
   
    * Install [MXNet 1.9.0](https://pypi.org/project/mxnet/1.9.0/) framework with CPU or GPU (preferred) support and [verify](https://mxnet.apache.org/versions/1.4.1/install/validate_mxnet.html) it:
    ```bash
    pip install mxnet-cuXXX==1.9.0
    ```

    * Run jupyter-lab:
    ```bash
    cd workspare
    jupyter-lab
    ```

### Preparation virtual environment

* Create virtual environment

```sh
    python3 -m venv venv
```

* To activate virtual environment

```sh
    source venv/bin/activate
```

* To deactivate virtual environment

```sh
    deactivate
```
