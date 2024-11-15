# Dogs vs. Cats tutorial based on [vit_pytorch](https://github.com/lucidrains/vit-pytorch) library

***Note: Please pay attention that this task uses the dataset from Kaggle. To get the dataset you
will need a Kaggle account and accept "Dogs vs. Cats" competition rules.***

Visual Transformers are gaining popularity among the Data Science community, so this tutorial is
intended to examine Visual Transformer behavior in Federated Learning setup.

## Installation of Kaggle API credentials

**Before the start please make sure that you installed sd_requirements.txt on your virtual
environment on an envoy machine.**

To use the [Kaggle API](https://github.com/Kaggle/kaggle-api), sign up for
a [Kaggle account](https://www.kaggle.com). Then go to the `'Account'` tab of your user
profile `(https://www.kaggle.com/<username>/account)` and select `'Create API Token'`. This will
trigger the download of `kaggle.json`, a file containing your API credentials. Place this file in
the location `cd ~/.kaggle/kaggle.json`

**Note: you will need to accept competition rules
at** https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition/rules

For your security, ensure that other users of your computer do not have read access to your
credentials. On Unix-based systems you can do this with the following command:

`chmod 600 ~/.kaggle/kaggle.json`

If you need proxy add "proxy": `"http://<ip_addr:port>" in kaggle.json`. It should looks like
that: `{"username":"your_username","key":"token", "proxy": "ip_addr:port"}`

*Information about Kaggle API settings has been taken from kagge-api readme. For more information
visit:* https://github.com/Kaggle/kaggle-api

*Useful link for a problem with proxy settings:* https://github.com/Kaggle/kaggle-api/issues/6

### Data

All information about the dataset you may find
on https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition/overview

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
5. Run the `director`: execute `start_director.sh` in director folder
6. Run the `envoys`: execute `start_envoy.sh` in each envoy folder. If kaggle-API setting are
   correct the download of the dataset will be started. If this is not the first `envoy` launch
   then the dataset will be redownloaded only if some part of the data are missing.
7. Run the [PyTorch_DogsCats_ViT.ipynb](workspace/PyTorch_DogsCats_ViT.ipynb) notebook using
   Jupyter lab in a prepared virtual environment. For more information about preparation virtual
   environment look [**
   Preparation virtual environment**](#preparation-virtual-environment)
   .
8. Congratulations! You've started your federated learning of Visual Transformer with OpenFL.

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