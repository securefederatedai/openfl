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
    AutoConfig,
)
from transformers.models.vit_mae.modeling_vit_mae import (
    ViTMAEDecoder,
    ViTMAEForPreTrainingOutput,
    ViTMAEEmbeddings,
    ViTMAEModel,
)
from transformers.modeling_outputs import ImageClassifierOutput, BackboneOutput
from typing import Optional, Union, Tuple
from src.modeling.LinearClassifier import LinearClassifier
from torch import Tensor, nn
from src.modeling.utils import HEAD_DICT
from src.modeling.MAEPretrainingFeatureExtractor import MAEPretrainingFeatureExtractor


class PretrainingModel(ViTMAEForPreTraining):
    def __init__(self, config: PretrainedConfig, **kwargs) -> None:
        super(ViTMAEForPreTraining, self).__init__(config)

        pretrain_config = kwargs.get("pretrain_config", None)
        if pretrain_config is None:
            pretrain_config = {"method": "mae_training"}

        if pretrain_config["method"] == "mae_training":
            decoder_config = AutoConfig.from_pretrained("facebook/vit-mae-base")
            decoder_config.update(self.config.to_dict())
            self.config = decoder_config
            self.feature_extractor: MAEPretrainingFeatureExtractor = MAEPretrainingFeatureExtractor(
                self.config
            )
            self.decoder = ViTMAEDecoder(
                self.config,
                num_patches=self.feature_extractor.feature_extractor.embeddings.patch_embeddings.num_patches,
            )

        else:
            raise ValueError(f"Unsupported pretraining method: {pretrain_config.method}")

    @property
    def vit(self):
        return self.feature_extractor
