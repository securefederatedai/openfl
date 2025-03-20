# %%
import torch
import matplotlib.pyplot as plt
from transformers import PreTrainedModel, Dinov2PreTrainedModel
from train import VitForSemanticSegmentation
from dataloader import create_dataset_dict, SEGMENT_CLASSES
import numpy as np

# Load the dataset
dataset_dict = create_dataset_dict()

# Load the trained model
#model = Dinov2ForSemanticSegmentation.from_pretrained("./results_unet", use_UNetDecoder=True)
model = VitForSemanticSegmentation.from_pretrained("./results_unet")

# Set the model to evaluation mode
model.eval()


# %%
# Function to visualize segmentation
def visualize_segmentation(image, label, prediction, or_image):
    fig, ax = plt.subplots(1, 4, figsize=(15, 5))
    image = (image - image.min()) / (image.max() - image.min())
    or_image = (or_image - or_image.min()) / (or_image.max() - or_image.min())
    ax[0].imshow(or_image)
    ax[0].set_title("or_image Image")
    ax[1].imshow(image.permute(1, 2, 0))
    ax[1].set_title("Original Image")
    ax[2].imshow(label)
    ax[2].set_title("Ground Truth")
    ax[3].imshow(prediction)
    ax[3].set_title("Prediction")
    plt.show()


# %%


# Visualize some examples

index = 0
#%%
count = 0
dataset_dict["train"] = dataset_dict["train"].shuffle()
while count < 10:
    sample = dataset_dict["train"][index]
    if sample["labels"][1:, :, :].sum() < 600:
        index += 1
        continue
    print(index)
    image = np.array(sample['flair'])
    pixel_values = sample["pixel_values"].unsqueeze(0)
    labels = sample["labels"].argmax(dim=0)
    count += 1
    index += 1
    print(labels.unique())

    with torch.no_grad():
        outputs = model(pixel_values)
        predictions = outputs.logits.argmax(dim=1).squeeze().cpu()

    visualize_segmentation(pixel_values.squeeze().cpu(), labels, predictions, image)

# %%

data = dataset_dict["train"][:5000]
# %%
images = np.stack([np.array(i) for i in data['flair']])

# %%
from matplotlib import pyplot as plt

maxes = images.max(axis=2).max(axis=1)

plt.hist(maxes, bins=100)
# %%
max_ss = np.percentile(maxes, 95)
images[max_ss>images].mean(), images[max_ss>images].std()
# %%
i = 10
plt.hist(images[i][images[i]>0].ravel(), bins=100)
# %%
