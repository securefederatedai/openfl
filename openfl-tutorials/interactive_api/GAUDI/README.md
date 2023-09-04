## OpenFL on Habana Gaudi AI training processors

Habana is an Intel company that develops advanced deep learning processors for training and inference. The Gaudi AI processor is desgined form the ground up to acceleate deep learning workloads. Habana's SynapseAI software suite  enables high performance, cost-efficient deep learning training and inference on Gaudi AI processors. Habana Gaudi is available on the cloud or on-premises. Check out  https://developer.habana.ai to learn more.
<br/>

## Enabling Federated Learning with OpenFL on Gaudi HPU


Gaudi devices are also referred to as Habana Processing Unit (HPU). The basic steps to run PyTorch or TensorFlow models on HPU are:

1) Setup the SynapseAI software (we recommend using the latest release). 
	- https://docs.habana.ai/en/latest/Installation_Guide/index.html
	

2) Update your model scripts to run on Gaudi 
	- https://docs.habana.ai/en/latest/PyTorch/Getting_Started_with_PyTorch_and_Gaudi/Getting_Started_with_PyTorch.html
	- https://docs.habana.ai/en/latest/TensorFlow/Migration_Guide/Porting_Simple_TensorFlow_Model_to_Gaudi.html

	**Note:** Remember to move the model to the device before initializing any optimizers. There is a dependency in the order of execution (moving model to HPU and intializing optimizer). 


	After making the changes, confirm your model runs on Gaudi.
<br/>

## Examples

In this directory, you will find examples of using OpenFL with Gaudi HPUs. (Note that only director and envoy workflow is currently tested with Gaudi.) 


- https://github.com/intel/openfl/tree/develop/openfl-tutorials/interactive_api/GAUDI/PyTorch_TinyImageNet:  workspace with a MobileNet-V2 PyTorch model that will download the Tiny-ImageNet dataset and train in a federation.


- https://github.com/intel/openfl/tree/develop/openfl-tutorials/interactive_api/GAUDI/PyTorch_Kvasir_UNet: workspace with a UNET PyTorch model that will download the Hyper-Kvasir dataset and train in a federation

	
- https://github.com/intel/openfl/tree/develop/openfl-tutorials/interactive_api/GAUDI/PyTorch_MedMNIST_2D: workspace with MedMNIST PyTorch model for 2D biomedical image classification that will download the BloodMNIST dataset and train in a federation






 