# %%
import argparse
import os

import evaluate
import numpy as np
import torch
from openfl.experimental.workflow.interface import (Aggregator, Collaborator,
                                                    FLSpec)
from openfl.experimental.workflow.placement import aggregator, collaborator
from openfl.experimental.workflow.runtime import LocalRuntime
from peft import LoraConfig, PeftModel, TaskType
from peft.config import PeftConfig
from torch.utils.tensorboard import SummaryWriter
from torchinfo import summary
from transformers import Trainer, TrainingArguments

os.chdir(PATH")
from src.dataloader import create_dataset_dict, SEGMENT_CLASSES, collate_fn
from src.model import VitForSemanticSegmentation
from src.utils import PeftModelForVit
from src.unet import UNet
from src.visionflow import FederatedFlow, set_writer
import matplotlib.pyplot as plt
from transformers import AutoModel, Dinov2Model


def visualize_segmentation(image, label, or_image, prediction=None):
    fig, ax = plt.subplots(1, 3 if prediction is None else 4, figsize=(15, 5))
    image = (image - image.min()) / (image.max() - image.min())
    or_image = (or_image - or_image.min()) / (or_image.max() - or_image.min())
    ax[0].imshow(or_image)
    ax[0].set_title("or_image Image")
    ax[1].imshow(image.permute(1, 2, 0))
    ax[1].set_title("Original Image")
    ax[2].imshow(label, cmap="gray")
    ax[2].set_title("Ground Truth")
    if prediction is not None:
        ax[3].imshow(prediction, cmap="gray")
        ax[3].set_title("Prediction")
    plt.show()


# %%
dataset_dicts, val_set = create_dataset_dict(patient_percentage=4, collaborator_count=4)

model_name = "facebook/dinov2-base"
model = VitForSemanticSegmentation(
    pretrained_model_name_or_path=model_name,
    id2label=SEGMENT_CLASSES,
    num_labels=len(SEGMENT_CLASSES),
    use_UNetDecoder=True,
    lora=True,
    dinov2=True,
)
lora_config = LoraConfig(
    task_type=TaskType.FEATURE_EXTRACTION,
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    target_modules="all-linear",
)
model.feature_extractor = PeftModelForVit(model.feature_extractor, lora_config)

# %%
from safetensors.torch import load_file

weights = load_file(
    PATH checkpoint-5/model.safetensors"
)
weights = {k.replace("feature_extractor.", ""): w for k, w in weights.items()}
feature_weights = {k: w for k, w in weights.items() if ("classifier" not in k)}
classifier_wights = {
    k.replace("classifier.", ""): w for k, w in weights.items() if ("classifier" in k)
}
model.feature_extractor.load_state_dict(feature_weights)
model.classifier.load_state_dict(classifier_wights)


training_args = TrainingArguments(
    output_dir="./results_testing",
    bf16=True,
    learning_rate=1e-4,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=10,
    weight_decay=0.01,
    remove_unused_columns=False,
    dataloader_num_workers=10,
    logging_steps=0.1,
    logging_strategy="steps",
    batch_eval_metrics=True,
    report_to=["tensorboard"],  # Add this line to enable TensorBoard logging
)
from transformers import Trainer

from src.utils import MetricAccumulator

metric = evaluate.load("mean_iou")
metric_accumulator = MetricAccumulator(metric, len(SEGMENT_CLASSES), ignore_index=None)
trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset_dicts[0]["train"],
            eval_dataset=dataset_dicts[0]["test"],
            data_collator=collate_fn,
            compute_metrics=metric_accumulator.compute,
        )

#%%
trainer.train()
#%%
eval_dict = trainer.evaluate()
print(eval_dict)
# %%
model.eval()
for n, sample in enumerate(dataset_dicts[0]["test"]):
    image = np.array(sample["flair"])
    pixel_values = sample["pixel_values"].unsqueeze(0)
    labels = sample["labels"].argmax(dim=0)

    with torch.no_grad():
        outputs = model(pixel_values.bfloat16().to("cuda"))
        predictions = outputs.logits.argmax(dim=1).squeeze().cpu()

    print(metric.compute(predictions=predictions, references=labels, num_labels=len(SEGMENT_CLASSES), ignore_index=None))
    print(metric.compute(predictions=predictions, references=labels, num_labels=len(SEGMENT_CLASSES), ignore_index=0))

    visualize_segmentation(pixel_values.squeeze().cpu(), labels, image, predictions)
    if n == 4:
        break
# %%
