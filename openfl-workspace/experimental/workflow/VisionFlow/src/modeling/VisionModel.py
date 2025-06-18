from typing import Union, Optional
from torch import Tensor, nn
import torch
from peft import get_peft_model_state_dict, set_peft_model_state_dict, PeftModel
from transformers import (
    PreTrainedModel,
    PretrainedConfig,
    AutoConfig,
    Dinov2Config,
    AutoModelForImageClassification,
    MODEL_FOR_BACKBONE_MAPPING,
)
from transformers.modeling_outputs import SemanticSegmenterOutput, ImageClassifierOutput
from peft import LoraConfig, TaskType, PeftType

from src.utils import PeftModelForVit, get_param_counts_log
from src.modeling.SegmentationModel import SemanticSegmentationModel
from src.modeling.ClassificationModel import ClassificationModel
from src.modeling.PretrainingModel import PretrainingModel
import logging

TASK_DICT = {
    "classification": ClassificationModel,
    "segmentation": SemanticSegmentationModel,
    "pretraining": PretrainingModel,
}
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VisionModel(nn.Module):
    def __init__(
        self,
        task_type: str = "classification",
        name_or_path=None,
        use_peft: bool = False,
        fine_tune_backbone: bool = False,
        lora_config={},
        pretrained_model_path=None,
        **model_config_kwargs,
    ):
        super().__init__()
        self.task_type = task_type
        if name_or_path is None:
            name_or_path = "facebook/dinov2-base"
        config: PretrainedConfig = AutoConfig.from_pretrained(name_or_path)

        model_config_kwargs.update({"output_hidden_states": True, "interpolate_pos_encoding": True})
        default_model_config = {}
        default_model_config.update(model_config_kwargs)
        config.update(model_config_kwargs)
        # Determine if config is a backbone
        self.model: Union[
            SemanticSegmentationModel, ClassificationModel, PeftModel, PretrainingModel
        ] = TASK_DICT[task_type.lower()](config=config, **model_config_kwargs)

        if hasattr(self.model, "num_labels"):
            self.num_labels = self.model.num_labels

        if not fine_tune_backbone:
            for name, param in self.model.feature_extractor.named_parameters():
                param.requires_grad = False

        if use_peft:
            if not isinstance(config, Dinov2Config):
                raise ValueError(
                    "PEFT is only supported for Dinov2 backbone in this implementation."
                )
            default_lora_config = {
                "task_type": TaskType.FEATURE_EXTRACTION,
                "r": 4,
                "lora_alpha": 8,
                "lora_dropout": 0.1,
                "target_modules": "all-linear",
                "modules_to_save": ["head"],
            }
            default_lora_config.update(lora_config)
            self.model = PeftModelForVit(self.model, LoraConfig(**default_lora_config))
            self.model.print_trainable_parameters()
        self.using_peft = use_peft

        if pretrained_model_path:
            logger.info(f"Loading pretrained model from {pretrained_model_path}")
            pretrained_model_weights = torch.load(pretrained_model_path, map_location="cpu")
            missing_keys = []
            expected_keys = set(pretrained_model_weights.keys())
            temp_keys = []
            for name, param in self.named_parameters():
                current_name = name.replace(
                    "feature_extractor.", "feature_extractor.feature_extractor."
                )
                if pretrained_model_weights.get(current_name) is not None:
                    param.data = pretrained_model_weights.get(current_name)
                    temp_keys.append(current_name)
                else:
                    missing_keys.append(name)
            unexpected_keys = expected_keys - set(temp_keys)
            logger.info(
                f"Loaded pretrained weights with {len(missing_keys)} missing keys and {len(unexpected_keys)} unexpected keys."
            )

        logger.info(get_param_counts_log(self.model))

    def get_weights(self):
        """Return a single dict containing all trainable weights."""
        if self.using_peft:
            weights = get_peft_model_state_dict(self.model)
        else:
            weights = self.model.state_dict()
        return weights

    def set_weights(self, weights):
        """Load weights from a single dict."""
        if self.using_peft and "peft" in weights:
            set_peft_model_state_dict(self.model, weights)
        else:
            self.model.load_state_dict(weights, strict=False)

    def forward(
        self, **kwargs: Union[Tensor, dict, str]
    ) -> Union[tuple, SemanticSegmenterOutput, ImageClassifierOutput]:
        if self.task_type == "classification":
            return self.model(**kwargs)
        elif self.task_type == "segmentation":
            return self.model(**kwargs)
        elif self.task_type == "pretraining":
            return self.model(
                interpolate_pos_encoding=self.model.config.interpolate_pos_encoding, **kwargs
            )
