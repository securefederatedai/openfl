<!-- ABOUT THE PROJECT -->
## About The Project

This is an example using OpenFL for Segmentation on the BrainMRI Segmentation medical decathlon dataset.

Here's why:
* There is another example available with similar task, but the dataset handled is small. This time, the dataset is huge.
* Organize large dataset and set it up to use with OpenFL


<p align="right">(<a href="#top">back to top</a>)</p>



<!-- GETTING STARTED -->
## Getting Started

This is a federated learning experiment therefore, you need to run the director and the envoy. Right now, we have one envoy present in this folder. To increase the scope of this experiment (to multiple envoy setup) just replicate the folder and change the settings like the environment and world_size (see envoy/envoy_config.yaml) and the BrainMRI2D_Segmentation_OpenFL_Tutorial.ipynp for more info.
 
## Modifying the specifics / Understanding the code

The shard_descriptor.py file in envoy/ contains the code for reading the data on the envoy side. This is where we perform augmentations and a lot of other stuff. It is worth checking out.

Other changes concerning hyperparameters of the experiment can be tuned / changed in the workspace/BrainMRI2D_Segmentation_OpenFL_Tutorial.ipynb file.

**NOTE**: It is important to note that in actual experiments, the director and envoy nodes will be running on different machines. For that, we will have to know the FQDN (Fully qualified domain name) of the Director Node and the port on which it runs, and subsequently modify the director/director.yaml file and the envoy/start_envoy.sh file. Basically, update the Director FQDN and port everywhere. Rest, the tutorial will run out of the box.

<p align="right">(<a href="#top">back to top</a>)</p>


## Usage

To run this tutorial, you need to initially launch the director node. This can be done with the command 
```sh
. ./director/start_director.sh
```


Similarly, we then start the two envoy nodes with the help of shell scripts given in the repository

```sh
. ./envoy/start_envoy.sh
```

**NOTE** : The dataset is huge and therefore, the envoy will take 10-15 minutes to start the first time.

Now, you can move on to the workspace/BrainMRI2D_Segmentation_OpenFL_Tutorial.ipynb file. Where you can launch experiments and track the training.

<p align="right">(<a href="#top">back to top</a>)</p>