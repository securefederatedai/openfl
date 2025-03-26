# %%
from torchinfo import summary
from peft import LoraConfig, TaskType
from src.dataloader import SEGMENT_CLASSES
from src.utils import PeftModelForVit
from src.unet import UNet
from src.model import VitForSemanticSegmentation


def get_model(use_vit, use_dino, use_decoder_unet, use_lora):
    if use_vit:
        if use_dino:
            model_name = "facebook/dinov2-base"
        else:
            model_name = "google/vit-base-patch16-224"
        model = VitForSemanticSegmentation(
            pretrained_model_name_or_path=model_name,
            id2label=SEGMENT_CLASSES,
            num_labels=len(SEGMENT_CLASSES),
            use_UNetDecoder=use_decoder_unet,
            lora=use_lora,
            dinov2=use_dino,
        )
        for name, param in model.named_parameters():
            if name.startswith("feature_extractor"):
                param.requires_grad = False
        if use_lora:
            lora_config = LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION,
                r=8,
                lora_alpha=16,
                lora_dropout=0.1,
                target_modules="all-linear",
            )
            model.feature_extractor = PeftModelForVit(model.feature_extractor, lora_config)
            model.feature_extractor.print_trainable_parameters()
        else:
            summary(model.feature_extractor, input_size=(1, 3, 224, 224))
        patches = 224 // model.feature_extractor.config.patch_size
        embeddings = model.feature_extractor.config.hidden_size
        summary(
            model.classifier,
            input_size=(5 if use_decoder_unet else 1, patches * patches, embeddings),
        )
    else:
        model = UNet(3, len(SEGMENT_CLASSES))
        summary(model, input_size=(1, 3, 224, 224))
    return model
