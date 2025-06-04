#%%
import logging

import evaluate
import torch
from openfl.experimental.workflow.interface import FLSpec
from openfl.experimental.workflow.placement import aggregator, collaborator
from transformers import TrainingArguments
from datasets import Dataset

from src.Brats2020_dataloader import SEGMENT_CLASSES, collate_fn
from src.utils import FedAvg, FederatedTrainer, Metric_Computer
from src.modeling.VisionModel import VisionModel
from peft import get_peft_model_state_dict, set_peft_model_state_dict, PeftModel

#%%
use_peft = True
model = VisionModel(
                task_type='classification',
                use_peft=use_peft,
                model_config_kwargs={"num_labels": 10, "image_size": 224}
            )
# %%
# %%
weights = model.model.head.state_dict()
if model.using_peft:
    weights.update(get_peft_model_state_dict(model.model))
from copy import deepcopy
weights = deepcopy(weights)
# %%
weights['base_model.model.feature_extractor.encoder.layer.0.attention.attention.key.lora_A.weight'][0,0] = 999
weights['classifier.weight'][0,0,0,0] = 999
#%%
model.model.head.state_dict()['classifier.weight']
#%%
get_peft_model_state_dict(model.model)['base_model.model.feature_extractor.encoder.layer.0.attention.attention.key.lora_A.weight']
# %%
model.model.head.load_state_dict(weights, strict=False)
set_peft_model_state_dict(model.model, weights)
# %%
