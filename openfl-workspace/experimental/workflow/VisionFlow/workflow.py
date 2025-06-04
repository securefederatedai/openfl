# %%
# %%
import os

os.chdir(os.path.dirname(__file__))

from openfl.experimental.workflow.interface import Aggregator, Collaborator
from openfl.experimental.workflow.runtime import LocalRuntime


from src.Brats2020_dataloader import create_dataset_dict
from src.VisionFlow import VisionFlow
from datasets import load_dataset
from datasets import Dataset, DatasetDict, Image


# Load CIFAR-10 dataset


from src.Brats2020_dataloader import IMAGE_TYPES
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter(log_dir="./output/tensorboard_logs")

collaborator_names = ["Portland", "Seattle", "Chandler", "Bangalore"]
collaborator_names = ["Portland", "Seattle"]

task = "classification"
if task == "classification":
    from src.dataset.img_classification import prepare_data_for_image_classification

    dataset_name = "Falah/Alzheimer_MRI"
    dataset = load_dataset(dataset_name)

    label_feature = "label"
    image_feature = "image"
    image_size = 224

    dataset_dicts, number_of_labels = prepare_data_for_image_classification(
        dataset,
        collaborator_count=len(collaborator_names),
        non_iid=True,
        image_feature=image_feature,
        label_feature=label_feature,
        image_size=image_size,
        debug_size=True,
    )
    dataset_dicts[0]["train"][0]["image"]
elif task == "segmentation":
    patient_count = 10
    dataset_dicts, val_set = create_dataset_dict(
        dataset_path="/home/omar/Documents/mine/INTEL/datasets/Processed_TrainingData/",
        number_of_patients_per_collaborator=patient_count,
        collaborator_count=4,
    )

# %%
training_args = {
    "bf16": True,
    "output_dir": "./output",
    "per_device_train_batch_size": 32,
    "per_device_eval_batch_size": 32,
    "num_train_epochs": 1,
    "weight_decay": 0.01,
    "logging_steps": 0.1,
    "logging_strategy": "steps",
    "dataloader_num_workers": 1,
    "batch_eval_metrics": True,
    "remove_unused_columns": False,
}

my_aggregator = Aggregator()
my_aggregator.private_attributes = {}


collaborators = [Collaborator(name=name) for name in collaborator_names]
for idx, current_collaborator in enumerate(collaborators):
    # Set the private attributes of the Collaborator to include their specific training and testing data loaders
    train_dataset = dataset_dicts[idx]["train"]
    eval_dataset = dataset_dicts[idx]["test"]

    current_collaborator.private_attributes = {
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
    }

local_runtime = LocalRuntime(
    aggregator=my_aggregator, collaborators=collaborators, backend="single_process"
)
model_config_kwargs = {"num_labels": number_of_labels, "image_size": image_size}
use_peft = True
if True:
    # model_config_kwargs.update({"name_or_path": "google/vit-base-patch16-224"})
    # model_config_kwargs.update({"name_or_path": "facebook/convnext-base-224-22k-1k"})
    # model_config_kwargs.update({"name_or_path": "microsoft/swin-base-patch4-window7-224-in22k"})
    # model_config_kwargs.update({"name_or_path": "facebook/vit-mae-base"})

    # model_config_kwargs.update({"name_or_path": "microsoft/resnet-50"})
    model_config_kwargs.update({"head": "MLPHead", "head_kwargs": {"hidden_layers": [512, 256]}})
    use_peft = False
    pass
# %%
flflow = VisionFlow(
    rounds=3,
    task_type="classification",
    global_validation_dataset=None,
    training_args=training_args,
    use_peft=use_peft,
    model_config_kwargs=model_config_kwargs,
    move_to_cpu_end_of_training=True,
    writer=writer,
)
flflow.runtime = local_runtime
flflow.run()

# %%
