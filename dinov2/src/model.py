# %%
import torch
import os
from transformers import Dinov2Model, ViTModel
from transformers.modeling_outputs import SemanticSegmenterOutput
import torch.nn as nn
from monai.losses import DiceLoss
from peft import get_peft_model_state_dict, set_peft_model_state_dict
from torchinfo import summary
from peft import LoraConfig, TaskType
from src.dataloader import SEGMENT_CLASSES
from src.utils import PeftModelForVit
from src.unet import UNet


os.chdir(PATH)
from src.heads import LinearClassifierToken, UNetDecoder


class VitForSemanticSegmentation(nn.Module):
    def __init__(
        self, use_UNetDecoder=False, output_hidden_states=True, lora=False, dinov2=True, **kwargs
    ):
        super(VitForSemanticSegmentation, self).__init__()
        self.lora = lora
        if dinov2:
            self.feature_extractor = Dinov2Model.from_pretrained(**kwargs)
        else:
            self.feature_extractor = ViTModel.from_pretrained(**kwargs)
        self.config = self.feature_extractor.config

        self.use_UNetDecoder = use_UNetDecoder
        self.patch_size = self.config.patch_size
        if use_UNetDecoder:
            self.classifier = UNetDecoder(
                self.config.hidden_size, out_channels=self.config.num_labels
            )
        else:
            patches = 224 // self.config.patch_size
            self.classifier = LinearClassifierToken(
                self.config.hidden_size, patches, patches, self.config.num_labels
            )
        self.output_hidden_states = output_hidden_states

    def get_weights(self):
        if self.lora:
            return get_peft_model_state_dict(self.feature_extractor), self.classifier.state_dict()
        else:
            return self.classifier.state_dict()

    def set_weights(self, weights):
        if self.lora:
            peft_weights, classifier_weights = weights
            set_peft_model_state_dict(self.feature_extractor, peft_weights)
            self.classifier.load_state_dict(classifier_weights)
        else:
            self.classifier.load_state_dict(weights)

    def forward(self, pixel_values, output_attentions=False, labels=None):
        # use frozen features
        outputs = self.feature_extractor(
            pixel_values,
            output_hidden_states=self.output_hidden_states,
            output_attentions=output_attentions,
        )
        # get the patch embeddings - so we exclude the CLS token
        patch_embeddings = outputs.last_hidden_state[:, 1:, :]

        # get skip connections from the transformer model
        if self.use_UNetDecoder:
            skip_connections = outputs.hidden_states[-4:]
            patch_embeddings = [i[:, 1:, :] for i in skip_connections] + [patch_embeddings]
        # convert to logits and upsample to the size of the pixel values
        logits = self.classifier(patch_embeddings)
        logits = torch.nn.functional.interpolate(
            logits, size=pixel_values.shape[2:], mode="bilinear", align_corners=False
        )

        loss = None
        if labels is not None:
            # important: we're going to use 0 here as ignore index instead of the default -100
            # as we don't want the model to learn to predict background
            if True:
                # loss_fct = DiceLoss(softmax=True, include_background=False)
                loss_fct = DiceLoss(softmax=True)
                loss = loss_fct(logits, labels)
            else:
                loss_fct = torch.nn.CrossEntropyLoss(
                    ignore_index=0,
                )
                loss = loss_fct(logits, labels.argmax(dim=1))

        return SemanticSegmenterOutput(
            loss=loss,
            logits=logits,
            hidden_states=None,
            attentions=None,
        )
