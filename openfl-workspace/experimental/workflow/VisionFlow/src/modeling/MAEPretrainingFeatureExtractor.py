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
from transformers.models.vit_mae.modeling_vit_mae import (
    ViTMAEDecoder,
    ViTMAEForPreTrainingOutput,
    ViTMAEEmbeddings,
    ViTMAEModel,
    ViTMAEModelOutput,
    BaseModelOutput,
    ViTMAEPreTrainedModel,
)
from transformers.modeling_outputs import BackboneOutput, ImageClassifierOutput
from typing import Optional, Union, Tuple
from src.modeling.LinearClassifier import LinearClassifier
from torch import Tensor, nn
from src.modeling.utils import HEAD_DICT


class MAEPretrainingFeatureExtractor(ViTMAEPreTrainedModel):
    def __init__(self, config: PretrainedConfig, **kwargs) -> None:
        super().__init__(config)

        if config.model_type in MODEL_FOR_BACKBONE_MAPPING._model_mapping:
            self.feature_extractor: AutoBackbone = AutoBackbone.from_pretrained(config.name_or_path)
        elif config.model_type in MODEL_FOR_PRETRAINING_MAPPING._model_mapping:
            self.feature_extractor = ViTForImageClassification.from_pretrained(
                config.name_or_path,
            ).vit
        else:
            self.feature_extractor = AutoModelForImageClassification.from_pretrained(
                config.name_or_path
            ).vit
            # self.feature_extractor.head = nn.Identity()

        pretrain_config = kwargs.get("pretrain_config", None)
        if pretrain_config is None:
            pretrain_config = {"method": "mae_training"}

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

        self.call_feature_extractor = call_feature_extractor

    def forward(
        self,
        pixel_values: Optional[Tensor] = None,
        head_mask: Optional[Tensor] = None,
        noise: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        interpolate_pos_encoding: bool = False,
    ) -> Union[tuple, ImageClassifierOutput]:
        output_attentions = (
            output_attentions if output_attentions is not None else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output, mask, ids_restore = self.embedding_forward(
            self.feature_extractor.embeddings,
            pixel_values,
            noise=noise,
            interpolate_pos_encoding=interpolate_pos_encoding,
        )

        encoder_outputs: BackboneOutput = self.mae_encoder_forward(
            self.feature_extractor.encoder,
            embedding_output,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        sequence_output = self.feature_extractor.layernorm(sequence_output)

        if not return_dict:
            return (sequence_output, mask, ids_restore) + encoder_outputs[1:]

        return ViTMAEModelOutput(
            last_hidden_state=sequence_output,
            mask=mask,
            ids_restore=ids_restore,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

    def mae_encoder_forward(
        self,
        encoder,
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ) -> Union[tuple, BaseModelOutput]:
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        for i, layer_module in enumerate(encoder.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None

            if encoder.gradient_checkpointing and encoder.training:
                layer_outputs = encoder._gradient_checkpointing_func(
                    layer_module.__call__,
                    hidden_states,
                    layer_head_mask,
                    output_attentions,
                )
            else:
                layer_outputs = layer_module(hidden_states, layer_head_mask, output_attentions)

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None
            )
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )

    def random_masking(self, embedding, sequence, noise=None):
        """
        Perform per-sample random masking by per-sample shuffling. Per-sample shuffling is done by argsort random
        noise.

        Args:
            sequence (`torch.LongTensor` of shape `(batch_size, sequence_length, dim)`)
            noise (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*) which is
                mainly used for testing purposes to control randomness and maintain the reproducibility
        """
        batch_size, seq_length, dim = sequence.shape
        len_keep = int(seq_length * (1 - self.config.mask_ratio))

        if noise is None:
            noise = torch.rand(batch_size, seq_length, device=sequence.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1).to(
            sequence.device
        )  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1).to(sequence.device)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        sequence_unmasked = torch.gather(
            sequence, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, dim)
        )

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([batch_size, seq_length], device=sequence.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return sequence_unmasked, mask, ids_restore

    def embedding_forward(
        self, embedding, pixel_values, noise=None, interpolate_pos_encoding: bool = False
    ):
        batch_size, num_channels, height, width = pixel_values.shape

        embeddings = embedding.patch_embeddings(pixel_values)
        if not interpolate_pos_encoding and (
            height != embedding.patch_embeddings.image_size[0]
            or width != embedding.patch_embeddings.image_size[1]
        ):
            raise ValueError(
                f"Input image size ({height}*{width}) doesn't match model ({self.image_size[0]}*{self.image_size[1]})."
            )
        if interpolate_pos_encoding:
            position_embeddings = embedding.interpolate_pos_encoding(embeddings, height, width)
        else:
            position_embeddings = embedding.position_embeddings

        # add position embeddings w/o cls token
        embeddings = embeddings + position_embeddings[:, 1:, :]

        # masking: length -> length * config.mask_ratio
        embeddings, mask, ids_restore = self.random_masking(embedding, embeddings, noise)

        # append cls token
        cls_token = embedding.cls_token + position_embeddings[:, :1, :]
        cls_tokens = cls_token.expand(embeddings.shape[0], -1, -1)
        embeddings = torch.cat((cls_tokens, embeddings), dim=1)

        return embeddings, mask, ids_restore
