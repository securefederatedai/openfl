import os
from skimage import io
import numpy as np

from openfl.utilities import validate_file_hash


def load_data():
    os.makedirs('data', exist_ok=True)
    os.system("wget -nc 'https://datasets.simula.no/hyper-kvasir/hyper-kvasir-segmented-images.zip'"
              " -O ./data/kvasir.zip")
    zip_sha384 = 'e30d18a772c6520476e55b610a4db457237f151e' \
                 '19182849d54b49ae24699881c1e18e0961f77642be900450ef8b22e7'
    validate_file_hash('./data/kvasir.zip', zip_sha384)
    os.system('unzip -n ./data/kvasir.zip -d ./data')


def read_data(image_path, mask_path):
    """
    Read image and mask from disk.
    """
    img = io.imread(image_path)
    assert (img.shape[2] == 3)
    mask = io.imread(mask_path)
    return img, mask[:, :, 0].astype(np.uint8)
