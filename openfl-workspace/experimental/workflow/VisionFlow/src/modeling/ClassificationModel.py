import torch
from transformers import (
    PretrainedConfig,
    AutoBackbone,
    AutoModelForPreTraining,
    MODEL_FOR_BACKBONE_MAPPING,
    MODEL_FOR_PRETRAINING_MAPPING,
    PreTrainedModel,
    Dinov2Backbone,
    AutoModelForImageClassification,
    ViTForImageClassification,
    ResNetBackbone,
    ViTMAEForPreTraining,
    ViTForImageClassification,
)
from transformers.modeling_outputs import SequenceClassifierOutput, BackboneOutput
from typing import Optional, Union
from src.modeling.LinearClassifier import LinearClassifier
from torch import Tensor, nn
from src.modeling.utils import HEAD_DICT


class ClassificationModel(PreTrainedModel):
    def __init__(self, config: PretrainedConfig, **kwargs) -> None:
        super().__init__(config)

        if config.model_type in MODEL_FOR_BACKBONE_MAPPING._model_mapping:
            self.feature_extractor: AutoBackbone = AutoBackbone.from_pretrained(config.name_or_path)
        elif config.model_type in MODEL_FOR_PRETRAINING_MAPPING._model_mapping:
            self.feature_extractor = ViTForImageClassification.from_pretrained(
                config.name_or_path,
            )
            self.feature_extractor.head = nn.Identity()
        else:
            self.feature_extractor = AutoModelForImageClassification.from_pretrained(
                config.name_or_path, config=config
            )
            self.feature_extractor.head = nn.Identity()

        head = kwargs.get("head", "LinearClassifier")
        head_kwargs = kwargs.get("head_kwargs", {})
        if head not in HEAD_DICT:
            if isinstance(head, nn.Module):
                self.head: nn.Module = head
            else:
                raise ValueError(f"Head {head} not found in HEAD_DICT")
        else:
            if self.feature_extractor.base_model_prefix in ["vit", "dinov2", "swin"]:
                self.head_in_channels = config.hidden_size
            else:
                self.head_in_channels = config.hidden_sizes[-1]
            self.head = HEAD_DICT[head](
                in_channels=self.head_in_channels, num_labels=config.num_labels, **head_kwargs
            )
        self.num_labels = config.num_labels

        if self.feature_extractor.base_model_prefix in ["vit", "dinov2", "swin"]:

            def call_feature_extractor(
                self,
                pixel_values: Tensor,
                output_attentions: Optional[bool] = None,
                output_hidden_states: Optional[bool] = None,
                return_dict: Optional[bool] = None,
            ) -> BackboneOutput:
                return self.feature_extractor(
                    pixel_values,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                )

            def get_backbone_output(outputs: BackboneOutput) -> Tensor:
                cls_token = outputs.hidden_states[-1][:, 0, :]  # Extract the CLS token
                cls_token = cls_token.reshape(-1, 1, 1, self.head_in_channels)
                cls_token = cls_token.permute(0, 3, 1, 2)
                return cls_token

        else:

            def call_feature_extractor(
                self,
                pixel_values: Tensor,
                output_attentions: Optional[bool] = None,
                output_hidden_states: Optional[bool] = None,
                return_dict: Optional[bool] = None,
            ) -> BackboneOutput:
                return self.feature_extractor(
                    pixel_values,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                )

            def get_backbone_output(outputs: BackboneOutput) -> Tensor:
                cls_token = outputs.hidden_states[-1]
                cls_token = cls_token.flatten(2)
                cls_token = torch.nn.functional.adaptive_avg_pool1d(cls_token, 1)
                cls_token = cls_token.unsqueeze(2)
                return cls_token

        self.call_feature_extractor = call_feature_extractor
        self.get_backbone_output = get_backbone_output

    def forward(
        self,
        pixel_values: Optional[Tensor] = None,
        head_mask: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs: Optional[Tensor],
    ) -> Union[tuple, SequenceClassifierOutput]:
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict: bool = return_dict if return_dict is not None else self.config.use_return_dict

        outputs: BackboneOutput = self.call_feature_extractor(
            self,
            pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # Use the CLS token for classification
        cls_token = self.get_backbone_output(outputs)

        logits = self.head(cls_token)
        logits = logits.squeeze(-1).squeeze(-1)

        loss = None
        if labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=None,
            attentions=None,
        )
