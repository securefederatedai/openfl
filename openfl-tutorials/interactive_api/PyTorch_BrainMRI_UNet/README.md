<!-- ABOUT THE PROJECT -->
## About The Project

This is an example using OpenFL for Segmentation on the BrainMRI Segmentation medical decathlon dataset.

Here's why:
* There is another example available with similar task, but the dataset handled is small. This time, the dataset is huge.
* Organize large dataset and set it up to use with OpenFL


<p align="right">(<a href="#top">back to top</a>)</p>



<!-- GETTING STARTED -->
## Getting Started

First of all, download the dataset that we will be using for this tutorial [GDriveLink](https://goo.gl/QzVZcm). Then download the `Task01_BrainTumour.tar` file from the Google Drive. <p>
Next, we process the data, for this run the `prepare_dataset.py` file. 

```sh
python prepare_dataset.py Task01_BrainTumour.tar
```

## Modifying the specifics / Understanding the code

The shard_descriptor.py file in envoy_1/ and envoy_2/ contains the code for reading the data on the envoy side. This is where we perform augmentations and a lot of other stuff. It is worth checking out.

Other changes concerning hyperparameters of the experiment can be tuned / changed in the workspace/BrainMRI2D_Segmentation_OpenFL_Tutorial.ipynb file.

**NOTE**: It is important to note that in actual experiments, the director and envoy nodes will be running on different machines. For that, we will have to know the FQDN (Fully qualified domain name) of the Director Node and the port on which it runs, and subsequently modify the director/director.yaml file and the envoy_1/start_envoy.sh and envoy_2/start_envoy.sh file. Basically, update the Director FQDN and port everywhere. Rest, the tutorial will run out of the box.

<p align="right">(<a href="#top">back to top</a>)</p>


## Usage

To run this tutorial, you need to initially launch the director node. This can be done with the command 
```sh
. ./director/start_director.sh
```


Similarly, we then start the two envoy nodes with the help of shell scripts given in the repository

```sh
. ./envoy_1/start_envoy.sh
```

```sh
. ./envoy_2/start_envoy.sh
```

Now, you can move on to the workspace/BrainMRI2D_Segmentation_OpenFL_Tutorial.ipynb file. Where you can launch experiments and track the training.

<p align="right">(<a href="#top">back to top</a>)</p>