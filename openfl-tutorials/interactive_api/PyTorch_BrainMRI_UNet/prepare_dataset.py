## THIS SCRIPT WILL PREPARE THE DATA FOR THIS TUTORIAL ##
## THE DATASET IS HUGE, SO IT WOULD TAKE SOME TIME TO PROCES IT ##
## ALSO, A LOT OF SPACE IS REQUIRED FOR THIS, SO KEEP ~70GB FEE ##

from email.mime import image
from logging import raiseExceptions
import os
import argparse
import tarfile
import json
import numpy as np
import random
import nibabel as nib
from tqdm import tqdm
import sys

parser = argparse.ArgumentParser("args")
parser.add_argument('path', help='Path to the tar file for medical decathlon Brain MRI dataset', type=str)
args = parser.parse_args()
data_path = args.path

# use tarfile to extract the data
tar = tarfile.open(data_path)
tar.extractall()
tar.close()

# Now read the data and process it to make the necessary images in the way we want it to be
data_folder = './Task01_BrainTumour/'
dataset_json_path = data_folder + 'dataset.json'
with open(dataset_json_path,'r') as f:
    dataset_json = json.load(f)

image_list = dataset_json['training']
num_images = len(image_list)
num_train = int(np.floor(num_images*.80))
train_image_list = image_list[:num_train]
valid_image_list = image_list[num_train:]

save_folder = './processed_data/'
if os.path.exists(save_folder):
    print("processed_data/ already exists, do you want to proceed? press y/Y to proceed and n/N to cancel")
    inp = input()
    if inp=='y' or inp=='Y':
        pass
    else:
        sys.exit("Exiting program")
else:
    os.mkdir(save_folder)


img_path_new = save_folder + 'img/'
if os.path.exists(img_path_new):
    print("processed_data/img/ already exists, do you want to proceed? press y/Y to proceed and n/N to cancel")
    inp = input()
    if inp=='y' or inp=='Y':
        pass
    else:
        sys.exit("Exiting program")
else:
    os.mkdir(img_path_new)


msk_path_new = save_folder + 'msk/'
if os.path.exists(msk_path_new):
    print("processed_data/msk/ already exists, do you want to proceed? press y/Y to proceed and n/N to cancel")
    inp = input()
    if inp=='y' or inp=='Y':
        pass
    else:
        sys.exit("Exiting program")
else:
    os.mkdir(msk_path_new)


train_data_csv = save_folder + 'train_data.csv'
valid_data_csv = save_folder + 'valid_data.csv'
with open(train_data_csv,'w') as f:
    f.write('name,img_path,msk_path\n')
with open(valid_data_csv,'w') as f:
    f.write('name,img_path,msk_path\n')


for t in tqdm(train_image_list):
    img_name = t['image'][t['image'].rfind('/')+1:t['image'].rfind('.nii')]
    img_path = os.path.join(data_folder, t['image'])
    msk_path = os.path.join(data_folder, t['label'])
    img = np.array(nib.load(img_path).dataobj)[:,:,:,0]
    msk = np.array(nib.load(msk_path).dataobj)
    num_slices = msk.shape[2]
    for slice in range(num_slices):
        img_slice = img[:,:,slice]
        msk_slice = msk[:,:,slice]
        img_slice_name = os.path.join(img_path_new,f'{img_name}_{slice}.npy')
        msk_slice_name = os.path.join(msk_path_new,f'{img_name}_{slice}.npy')
        np.save(img_slice_name,img_slice)
        np.save(msk_slice_name,msk_slice)
        with open(train_data_csv,'a') as f:
            f.write(f'{img_name},{img_slice_name},{msk_slice_name}\n')

for v in tqdm(valid_image_list):
    img_name = v['image'][v['image'].rfind('/')+1:v['image'].rfind('.nii')]
    img_path = os.path.join(data_folder, v['image'])
    msk_path = os.path.join(data_folder, v['label'])
    img = np.array(nib.load(img_path).dataobj)[:,:,:,0]
    msk = np.array(nib.load(msk_path).dataobj)
    num_slices = msk.shape[2]
    for slice in range(num_slices):
        img_slice = img[:,:,slice]
        msk_slice = msk[:,:,slice]
        img_slice_name = os.path.join(img_path_new,f'{img_name}_{slice}.npy')
        msk_slice_name = os.path.join(msk_path_new,f'{img_name}_{slice}.npy')
        np.save(img_slice_name,img_slice)
        np.save(msk_slice_name,msk_slice)
        with open(valid_data_csv,'a') as f:
            f.write(f'{img_name},{img_slice_name},{msk_slice_name}\n')

# Copy the dataset to the respective folders
os.system('cp -r processed_data envoy_1/')
os.system('cp -r processed_data envoy_2/')
os.system('cp processed_data/train_data.csv envoy_1/')
os.system('cp processed_data/train_data.csv envoy_2/')
os.system('cp processed_data/valid_data.csv envoy_1/')
os.system('cp processed_data/valid_data.csv envoy_2/')
