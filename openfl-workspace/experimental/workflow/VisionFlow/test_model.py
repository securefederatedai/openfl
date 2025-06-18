#%%
import logging
import os

os.chdir(os.path.dirname(__file__))
import evaluate
import torch
from openfl.experimental.workflow.interface import FLSpec
from openfl.experimental.workflow.placement import aggregator, collaborator
from transformers import TrainingArguments, AutoModelForImageClassification
from datasets import Dataset

from src.utils import FedAvg, FederatedTrainer, Metric_Computer
from src.modeling.VisionModel import VisionModel, TASK_DICT
from src.dataset.img_classification import image_classification_collate_fn, DEFAULT_LABEL_FEATURE
from src.dataset.img_pretraining import image_pretraining_collate_fn

#%%
task_type = 'classification'
use_peft = True
model_config_kwargs = {"num_labels": 4, "image_size": 224}
if False:
    model_config_kwargs.update(
                {"head": "MLPHead", "head_kwargs": {"hidden_layers": [512, 256]}}
            )
model = VisionModel(
                task_type=task_type,
                use_peft=use_peft,
                **model_config_kwargs,
            )

# %%
model
# %%
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
trainable_layers = [name for name, param in model.named_parameters() if param.requires_grad]
print("Trainable layers:", trainable_layers)
print(f"Trainable parameters: {trainable_params}")
# %%

model_2 = AutoModelForImageClassification.from_pretrained(
    "facebook/dinov2-base",
    num_labels=4,
    output_hidden_states=True,
)