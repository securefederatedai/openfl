from typing import Union, Optional
from torch import Tensor, nn
from peft import get_peft_model_state_dict, set_peft_model_state_dict, PeftModel
from transformers import PreTrainedModel, PretrainedConfig, AutoConfig, Dinov2Config
from transformers.modeling_outputs import SemanticSegmenterOutput
from peft import LoraConfig, TaskType

from src.utils import PeftModelForVit
from src.modeling.DinoV2Segmentation import SemanticSegmentationModel


class VisionModel(nn.Module):
    def __init__(
        self,
        task_type: str = "segmentation",
        name_or_path=None,
        use_peft: bool = False,
        image_size=224,
        num_labels=2,
        **kwargs
    ):
        super().__init__()
        self.task_type = task_type
        if name_or_path is None:
            name_or_path = "facebook/dinov2-base"
        config: PretrainedConfig = AutoConfig.from_pretrained(name_or_path, **kwargs)
        config.image_size = image_size
        config.num_labels = num_labels
        self.model: Union[SemanticSegmentationModel, PeftModel] = SemanticSegmentationModel(
            config=config
        )

        for name, param in self.model.feature_extractor.named_parameters():
            param.requires_grad = False

        if use_peft:
            lora_config = LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION,
                r=8,
                lora_alpha=16,
                lora_dropout=0.1,
                target_modules="all-linear",
            )
            self.model = PeftModelForVit(self.model, lora_config)
            self.model.print_trainable_parameters()
        self.using_peft = use_peft

    def get_weights(self):
        if self.using_peft:
            return get_peft_model_state_dict(self.model), self.model.head.state_dict()
        else:
            return self.model.head.state_dict()

    def set_weights(self, weights):
        if self.using_peft:
            peft_weights, classifier_weights = weights
            set_peft_model_state_dict(self.model, peft_weights)
            self.model.head.load_state_dict(classifier_weights)
        else:
            self.model.head.load_state_dict(weights)

    def forward(
        self,
        pixel_values: Optional[Tensor] = None,
        head_mask: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[tuple, SemanticSegmenterOutput]:
        return self.model(
            pixel_values=pixel_values,
            head_mask=head_mask,
            labels=labels,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
