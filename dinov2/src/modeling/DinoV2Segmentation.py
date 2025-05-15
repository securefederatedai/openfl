import torch
from monai.losses import DiceLoss
from transformers import (
    PretrainedConfig,
    AutoBackbone,
    PreTrainedModel,
    Dinov2Backbone
)
from transformers.modeling_outputs import SemanticSegmenterOutput, BackboneOutput
from typing import Optional, Union
from src.heads import LinearClassifier
from torch import Tensor


class SemanticSegmentationModel(PreTrainedModel):
    def __init__(self, config: PretrainedConfig) -> None:
        super().__init__(config)

        self.feature_extractor: AutoBackbone = AutoBackbone.from_pretrained(config.name_or_path)

        self.head: LinearClassifier = LinearClassifier(
            in_channels=config.hidden_size,
            tokenW=config.image_size // config.patch_size,
            tokenH=config.image_size // config.patch_size,
            num_labels=config.num_labels,
        )

    def forward(
        self,
        pixel_values: Optional[Tensor] = None,
        head_mask: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[tuple, SemanticSegmenterOutput]:
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict: bool = return_dict if return_dict is not None else self.config.use_return_dict

        outputs: BackboneOutput = self.feature_extractor(
            pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # get the patch embeddings - so we exclude the CLS token
        patch_embeddings = outputs.hidden_states[-1][:, 1:, :]

        logits = self.head(patch_embeddings)
        logits = torch.nn.functional.interpolate(
            logits, size=pixel_values.shape[2:], mode="bilinear", align_corners=False
        )

        loss = None
        if labels is not None:
            loss_fct = DiceLoss(softmax=True)
            loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SemanticSegmenterOutput(
            loss=loss,
            logits=logits,
            hidden_states=None,
            attentions=None,
        )
