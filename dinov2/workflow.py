# %%
import argparse
import os

import torch
from openfl.experimental.workflow.interface import Aggregator, Collaborator
from openfl.experimental.workflow.runtime import LocalRuntime
from torch.utils.tensorboard import SummaryWriter
from torchinfo import summary
from transformers import TrainingArguments

os.chdir("/home/omar/Documents/mine/INTEL/openfl/dinov2")
from src.Brats2020_dataloader import SEGMENT_CLASSES, create_dataset_dict
from src.unet import UNet
from src.VisionFlow import VisionFlow, set_writer

random_seed = 1
torch.manual_seed(random_seed)

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Federated Learning Workflow")
parser.add_argument("--lora", action="store_true", help="Use LoRA")
parser.add_argument("--use_dino", action="store_true", help="Use DINO model")
parser.add_argument("--use_decoder_unet", action="store_true", help="Use DINO model")
parser.add_argument("--use_vit", action="store_true", help="Use ViT model")
parser.add_argument("--fast", action="store_true", help="Enable fast mode")
parser.add_argument("--patient_percentage", type=float, default=0.1, help="Patient percentage")
parser.add_argument("--patient_count", type=int, default=0, help="Patient count")
parser.add_argument("--experiment", type=str, default="", help="Patient count")
args = parser.parse_args()

use_dino = args.use_dino
use_decoder_unet = args.use_decoder_unet
use_vit = args.use_vit
use_lora = args.lora
patient_percentage = args.patient_percentage
use_fast = args.fast
use_decoder_unet = args.use_decoder_unet
patient_count = args.patient_count
experiment = args.experiment

if patient_count > 0:
    dataset_dicts, val_set = create_dataset_dict(
        dataset_path="/home/omar/Documents/mine/INTEL/datasets/Processed_TrainingData/",
        patient_percentage=patient_count,
        collaborator_count=4,
    )
else:
    dataset_dicts, val_set = create_dataset_dict(
        dataset_path="/home/omar/Documents/mine/INTEL/datasets/Processed_TrainingData/",
        patient_percentage=patient_percentage,
        collaborator_count=4,
    )

if False:
    model = UNet(3, len(SEGMENT_CLASSES))
    model_type = "unet"
    summary(model, input_size=(1, 3, 224, 224))


# Format output directory based on options used
lora_type = "lora" if use_lora else "nolora"
head_type = "decoder" if use_decoder_unet else ""
split_type = (
    f"split:{str(patient_percentage).replace('.', '_')}"
    if patient_count == 0
    else f"count:{patient_count}"
)
dir_path = f"./{experiment}"
os.makedirs(dir_path, exist_ok=True)
model_type = "testetesded"
output_dir = f"{dir_path}/{model_type}_{lora_type}_{'decoder' if use_decoder_unet else ''}_{split_type}_{'_dummy' if use_fast else ''}"
os.makedirs(output_dir, exist_ok=True)

training_args = TrainingArguments(
    fp16=True,
    output_dir=output_dir,
    learning_rate=1e-4,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=1,
    weight_decay=0.01,
    logging_dir=f"{output_dir}/logs",
    logging_steps=0.1,
    logging_strategy="steps",
    remove_unused_columns=False,
    dataloader_num_workers=6,
    batch_eval_metrics=True,
    report_to=["tensorboard"],  # Add this line to enable TensorBoard logging
)


writer = SummaryWriter(log_dir=f"{output_dir}/metrics")
set_writer(writer)

# %%
# Setup participants
my_aggregator = Aggregator()
my_aggregator.private_attributes = {}

# Setup collaborators with private attributes
# collaborator_names = ['Portland', 'Seattle', 'Chandler','Bangalore']
collaborator_names = (
    ["Portland", "Seattle"] if use_fast else ["Portland", "Seattle", "Chandler", "Bangalore"]
)
collaborators = [Collaborator(name=name) for name in collaborator_names]
for idx, current_collaborator in enumerate(collaborators):
    # Set the private attributes of the Collaborator to include their specific training and testing data loaders
    train_dataset = dataset_dicts[idx]["train"]
    eval_dataset = dataset_dicts[idx]["test"]

    if use_fast:
        train_dataset = train_dataset.select(range(len(train_dataset) // 20))
        eval_dataset = eval_dataset.select(range(len(eval_dataset) // 20))

    current_collaborator.private_attributes = {
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
    }

local_runtime = LocalRuntime(
    aggregator=my_aggregator, collaborators=collaborators, backend="single_process"
)
print(f"Local runtime collaborators = {local_runtime.collaborators}")
# %%
flflow = VisionFlow(
    rounds=3 if use_fast else 10,
    global_validation_dataset=(
        val_set if not use_fast else val_set.select(range(len(eval_dataset) // 20))
    ),
    training_args=training_args,
    use_lora=use_lora,
    move_to_cpu_end_of_training=True,
    model_config_kwargs = {'output_hidden_states':True}
)
flflow.runtime = local_runtime
flflow.run()
# %%
torch.cuda.empty_cache()
