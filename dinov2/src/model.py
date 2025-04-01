# %%
import torch
import os
from transformers import Dinov2Model, ViTModel, Dinov2Config, PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import SemanticSegmenterOutput
import torch.nn as nn
from monai.losses import DiceLoss
from peft import get_peft_model_state_dict, set_peft_model_state_dict
from torchinfo import summary
from peft import LoraConfig, TaskType, PeftConfig
from src.dataloader import SEGMENT_CLASSES
from src.utils import PeftModelForVit
from src.unet import UNet


os.chdir(PATH)
from src.heads import LinearClassifierToken, UNetDecoder


class VitForSemanticSegmentation(nn.Module):
    def __init__(
        self,
        feature_extractor: PreTrainedModel = None,
        use_UNetDecoder=False,
        output_hidden_states=True,
        lora=False,
        dinov2=True,
        feature_extractor_config: PretrainedConfig = None,
        peft_config: PeftConfig = None,
        head: nn.Module = None,
        **kwargs
    ):
        super(VitForSemanticSegmentation, self).__init__()
        if feature_extractor_config is None:
            self.feature_extractor_config: Dinov2Config = Dinov2Config.from_pretrained(
                "facebook/dinov2-base"
            )

        if feature_extractor is None:
            self.feature_extractor: Dinov2Model = Dinov2Model.from_pretrained(
                self.feature_extractor_config
            )
        else:
            self.feature_extractor: PreTrainedModel = feature_extractor

        if peft_config is not None:
            self.using_peft = True
            self.feature_extractor: PeftModelForVit = PeftModelForVit(
                self.feature_extractor, peft_config
            )

        if head is not None:
            self.head = head
        else:
            self.patch_size = self.feature_extractor.config.patch_size
            patches = (
                self.feature_extractor.config.image_size // self.feature_extractor.config.patch_size
            )
            self.classifier = LinearClassifierToken(
                feature_extractor_config.hidden_size,
                patches,
                patches,
                feature_extractor_config.num_labels,
            )
        self.output_hidden_states = output_hidden_states

    def get_weights(self):
        if self.using_peft:
            return get_peft_model_state_dict(self.feature_extractor), self.classifier.state_dict()
        else:
            return self.classifier.state_dict()

    def set_weights(self, weights):
        if self.using_peft:
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
