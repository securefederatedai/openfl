import os
from skimage import io
import numpy as np
from openfl.utilities import validate_file_hash
import requests
from zipfile import ZipFile


def load_data():
    os.makedirs('data', exist_ok=True)
    with open('./data/kvasir.zip', 'wb') as zf:
        response = requests.get('https://datasets.simula.no/downloads/hyper-kvasir/'
                                'hyper-kvasir-segmented-images.zip')
        for chunk in response.iter_content(1024 * 1024 * 1024):
            zf.write(chunk)
    zip_sha384 = ('66cd659d0e8afd8c83408174'
                  '1ade2b75dada8d4648b816f2533c8748b1658efa3d49e205415d4116faade2c5810e241e')
    validate_file_hash('./data/kvasir.zip', zip_sha384)
    with ZipFile('./data/kvasir.zip') as f:
        f.extractall('./data')


def read_data(image_path, mask_path):
    """
    Read image and mask from disk.
    """
    img = io.imread(image_path)
    assert (img.shape[2] == 3)
    mask = io.imread(mask_path)
    return img, mask[:, :, 0].astype(np.uint8)
