# Copyright (C) 2020-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""You may copy this file as the starting point of your own model."""

import os

import logging
import nibabel as nib
import numpy as np
import numpy.ma as ma
from openfl.federated import TensorFlowDataLoader


logger = logging.getLogger(__name__)


class BratsDataloader(TensorFlowDataLoader):
    """TensorFlow Data Loader for the BraTS dataset."""

    def __init__(self, data_path, batch_size, percent_train=0.8, pre_split_shuffle=True, num_classes=1,
                 **kwargs):
        """Initialize.

        Args:
            data_path: The file path for the BraTS dataset
            batch_size (int): The batch size to use
            percent_train (float): The percentage of the data to use for training (Default=0.8)
            pre_split_shuffle (bool): True= shuffle the dataset before
            performing the train/validate split (Default=True)
            **kwargs: Additional arguments, passed to super init and load_from_nifti

        Returns:
            Data loader with BraTS data
        """
        super().__init__(batch_size, **kwargs)

        X_train, y_train, X_valid, y_valid = load_from_nifti(parent_dir=data_path,
                                                             percent_train=percent_train,
                                                             shuffle=pre_split_shuffle,
                                                             **kwargs)
        self.X_train = X_train
        self.y_train = y_train
        self.X_valid = X_valid
        self.y_valid = y_valid
        self.num_classes = num_classes


def train_val_split(features, labels, percent_train, shuffle):
    """Train/validation splot of the BraTS dataset.

    Splits incoming feature and labels into training and validation. The value
    of shuffle determines whether shuffling occurs before the split is performed.

    Args:
        features: The input images
        labels: The ground truth labels
        percent_train (float): The percentage of the dataset that is training.
        shuffle (bool): True = shuffle the dataset before the split

    Returns:
        train_features: The input images for the training dataset
        train_labels: The ground truth labels for the training dataset
        val_features: The input images for the validation dataset
        val_labels: The ground truth labels for the validation dataset
    """

    def split(lst, idx):
        """Split a Python list into 2 lists.

        Args:
            lst: The Python list to split
            idx: The index where to split the list into 2 parts

        Returns:
            Two lists

        """
        if idx < 0 or idx > len(lst):
            raise ValueError('split was out of expected range.')
        return lst[:idx], lst[idx:]

    nb_features = len(features)
    nb_labels = len(labels)
    if nb_features != nb_labels:
        raise RuntimeError('Number of features and labels do not match.')
    if shuffle:
        new_order = np.random.permutation(np.arange(nb_features))
        features = features[new_order]
        labels = labels[new_order]
    split_idx = int(percent_train * nb_features)
    train_features, val_features = split(lst=features, idx=split_idx)
    train_labels, val_labels = split(lst=labels, idx=split_idx)
    return train_features, train_labels, val_features, val_labels


def load_from_nifti(parent_dir,
                    percent_train,
                    shuffle,
                    channels_last=True,
                    task='whole_tumor',
                    **kwargs):
    """Load the BraTS dataset from the NiFTI file format.

    Loads data from the parent directory (NIfTI files for whole brains are
    assumed to be contained in subdirectories of the parent directory).
    Performs a split of the data into training and validation, and the value
    of shuffle determined whether shuffling is performed before this split
    occurs - both split and shuffle are done in a way to
    keep whole brains intact. The kwargs are passed to nii_reader.

    Args:
        parent_dir: The parent directory for the BraTS data
        percent_train (float): The percentage of the data to make the training dataset
        shuffle (bool): True means shuffle the dataset order before the split
        channels_last (bool): Input tensor uses channels as last dimension (Default is True)
        task: Prediction task (Default is 'whole_tumor' prediction)
        **kwargs: Variable arguments to pass to the function

    Returns:
        train_features: The input images for the training dataset
        train_labels: The ground truth labels for the training dataset
        val_features: The input images for the validation dataset
        val_labels: The ground truth labels for the validation dataset

    """
    path = os.path.join(parent_dir)
    subdirs = os.listdir(path)
    subdirs.sort()
    if not subdirs:
        raise SystemError(f'''{parent_dir} does not contain subdirectories.
Please make sure you have BraTS dataset downloaded
and located in data directory for this collaborator.
        ''')
    subdir_paths = [os.path.join(path, subdir) for subdir in subdirs]

    imgs_all = []
    msks_all = []
    for brain_path in subdir_paths:
        these_imgs, these_msks = nii_reader(
            brain_path=brain_path,
            task=task,
            channels_last=channels_last,
            **kwargs
        )
        # the needed files where not present if a tuple of None is returned
        if these_imgs is None:
            logger.debug(f'Brain subdirectory: {brain_path} did not contain the needed files.')
        else:
            imgs_all.append(these_imgs)
            msks_all.append(these_msks)

    # converting to arrays to allow for numpy indexing used during split
    imgs_all = np.array(imgs_all)
    msks_all = np.array(msks_all)

    # note here that each is a list of 155 slices per brain, and so the
    # split keeps brains intact
    imgs_all_train, msks_all_train, imgs_all_val, msks_all_val = train_val_split(
        features=imgs_all,
        labels=msks_all,
        percent_train=percent_train,
        shuffle=shuffle
    )
    # now concatenate the lists
    imgs_train = np.concatenate(imgs_all_train, axis=0)
    msks_train = np.concatenate(msks_all_train, axis=0)
    imgs_val = np.concatenate(imgs_all_val, axis=0)
    msks_val = np.concatenate(msks_all_val, axis=0)

    return imgs_train, msks_train, imgs_val, msks_val


def parse_segments(seg, msk_modes):
    """Parse the label segments.

    Each channel corresponds to a different region of the tumor, decouple and stack these

    mode_to_key_value = {"necrotic": 1, "edema": 2, "GD": 4}

    Args:
        seg: The segmentation labels
        msk_modes: Label mode to parse for the model

    Returns:
        The processed mask labels

    """
    msks_parsed = []
    for slice_ in range(seg.shape[-1]):
        # which mask values indicicate which label mode
        mode_to_key_value = {'necrotic': 1, 'edema': 2, 'GD': 4}
        curr = seg[:, :, slice_]
        this_msk_parts = []
        for mode in msk_modes:
            this_msk_parts.append(
                ma.masked_not_equal(curr, mode_to_key_value[mode]).filled(
                    fill_value=0
                )
            )
        msks_parsed.append(np.dstack(this_msk_parts))

    # Replace all tumorous areas with 1 (previously marked as 1, 2 or 4)
    mask = np.asarray(msks_parsed)
    mask[mask > 0] = 1

    return mask


def normalize_stack(imgs):
    """Z-score normalization of the input images.

    Args:
        imgs: The input images

    Returns:
        The processed (normalized) input images

    """
    imgs = (imgs - np.mean(imgs)) / (np.std(imgs))
    return imgs


def resize_data(dataset, new_size=128, rotate=3):
    """Resize 2D images within data by cropping equally from their boundaries.

    Args:
        dataset (np.array): Data containing 2D images whose dimensions are
        along the 1st and 2nd axes.
        new_size (int): Dimensions of square image to which resizing will
        occur. Assumed to be an even distance away from both dimensions of
        the images within dataset. (default 128) rotate (int): Number of
        counter clockwise 90 degree rotations to perform.

    Returns:
        (np.array): resized data

    Raises:
        ValueError: If (dataset.shape[1] - new_size) and
         (dataset.shape[2] - new_size) are not both even integers.

    """
    # Determine whether dataset and new_size are compatible with existing logic
    if (dataset.shape[1] - new_size) % 2 != 0 and (dataset.shape[2] - new_size) % 2 != 0:
        raise ValueError(f'dataset shape: {dataset.shape} and new_size: {new_size} '
                         f'are not compatible with existing logic')

    start_index = int((dataset.shape[1] - new_size) / 2)
    end_index = dataset.shape[1] - start_index

    if rotate != 0:
        resized = np.rot90(dataset[:, start_index:end_index, start_index:end_index],
                           rotate, axes=(1, 2))
    else:
        resized = dataset[:, start_index:end_index, start_index:end_index]

    return resized


# adapted from https://github.com/NervanaSystems/topologies
def _update_channels(imgs, msks, img_channels_to_keep,
                     msk_channels_to_keep, channels_last):
    """Filter the channels of images and move placement of channels in shape if desired.

    Args:
        imgs (np.array): A stack of images with channels (channels could be
        anywhere in number from one to four. Images are indexed along first
        axis, with channels along fourth (last) axis. msks (np.array): A stack
        of binary masks with channels (channels could be anywhere in number
        from one to four. Images are indexed along first axis, with channels
        along fourth (last) axis. img_channels_to_keep (flat np.ndarray): the
        channels to keep in the image (remove others) msk_channels_to_keep
        (flat np.ndarray): the channels to sum in the mask (resulting in
        a single channel array)
        channels_last (bool): Return channels in last axis? otherwise just
         after first

    Returns:
        images, masks with selected channels
    """
    new_imgs = imgs[:, :, :, img_channels_to_keep]
    # the mask channels that are kept are summed over to leave one channel
    # note the indices producing non-zero entries on these masks are mutually exclusive
    # so that the result continues to be an array with only ones and zeros
    msk_summands = [
        msks[:, :, :, channel:channel + 1] for channel in msk_channels_to_keep
    ]
    new_msks = np.sum(msk_summands, axis=0)

    if not channels_last:
        new_order = [0, 3, 1, 2]
        return np.transpose(new_imgs, new_order), np.transpose(new_msks, new_order)
    else:
        return new_imgs, new_msks


def list_files(root, extension, parts):
    """Construct files from root, parts."""
    files = [root + part + extension for part in parts]
    return files


def nii_reader(brain_path, task, channels_last=True,
               numpy_type='float64', normalization='by_mode', **kwargs):
    """Fetch a whole brain 3D image from disc.

    Assumes data_dir contains only subdirectories, each containing exactly one
    of each of the following files: "<subdir>_<type>.nii.gz", where <subdir> is
    the name of the subdirectory and <type> is one of ["t1", "t2","flair","t1ce"],
    as well as a segmentation label file "<subdir>_<suffix>", where suffix is:
    'seg_binary.nii.gz', 'seg_binarized.nii.gz', or 'SegBinarized.nii.gz'.
    These files provide all modes and segmenation label for a single patient
    brain scan consisting of 155 axial slice images. The reader returns the whole
    brain (all 155 slices).
    Note: Much of the logic here is included in order to allow for the reader
    to have the same functionality (shared code) regardless of normalization
    method. This allows us to validate all functionality except the normalization,
    against data on disc that was previously processed by normalizing all scanning
    modalities together. Post validation as desired, the code can be simplified.

    Args:
        brain_path (str): path to files containing image features and label
        set for a single brain.
        task (string): Describes the classification task, which determines
        the way in which the scanning modes and label information is filtered
        and combined.
        channels_last (bool): Determines whether reader should output channels
        in the position of the last axis.Otherwise channels are placed just
        after first axis.
        numpy_type (string): The numpy datatype for final casting before return.
        normalization (string): Determines whether the feature scanning modes
        are not normalized (None), normalized by scanning mode ('by_mode'),
        or normalized across all modes ('modes_together').
        **kwargs: unused

    Returns:
        np.array: whole 3D brain associated to a subdirectory given by the index

    Raises:
        ValueError: If label_type is not in
                    ['whole_tumor', 'enhanced_tumor', 'active_core', 'other']
        ValueError: If the path determined by idx and indexed_data_paths points
                    to a file with incomplete data

    """
    files = os.listdir(brain_path)
    # link task to appropriate image and mask channels of interest
    img_modes = ['t1', 't2', 'flair', 't1ce']
    msk_modes = ['necrotic', 'edema', 'GD']
    task_to_img_modes = {
        'whole_tumor': ['flair'],
        'enhanced_tumor': ['t1'],
        'active_core': ['t2'],
        'other': ['t1', 't2', 'flair', 't1ce'],
    }
    task_to_msk_modes = {
        'whole_tumor': ['necrotic', 'edema', 'GD'],
        'enhanced_tumor': ['GD'],
        'active_core': ['edema', 'GD'],
        'other': ['necrotic', 'edema', 'GD'],
    }
    msk_names = ['seg_binary', 'seg_binarized', 'SegBinarized', 'seg']

    # validate that task is an allowed key
    if task not in task_to_img_modes.keys():
        raise ValueError(f'{task} is not a valid task')

    # validate that the tasks used in task_to_img_modes and
    # task_to_msk_modes are the same
    if set(task_to_img_modes.keys()) != set(task_to_msk_modes.keys()):
        raise RuntimeError('Hard coded keys to task_to_img_modes'
                           'and task_to_mask_modes are not the same and should be.')

    # check that all appropriate files are present
    file_root = brain_path.split('/')[-1] + '_'
    extension = '.nii.gz'

    # record files needed
    # needed mask files are currntly independent of task
    need_files_oneof = list_files(file_root, extension, msk_names)
    if normalization != 'modes_together':
        need_files_all = list_files(file_root, extension, task_to_img_modes[task])
    else:
        need_files_all = list_files(file_root, extension, img_modes)

    correct_files = np.all([
        (reqd in files)
        for reqd in need_files_all
    ]) and np.sum([
        (reqd in files)
        for reqd in need_files_oneof
    ]) == 1

    if not correct_files:
        return None, None

    # get image (features)
    imgs_per_mode = []
    for file in need_files_all:
        path = os.path.join(brain_path, file)
        full_brain = np.array(nib.load(path).dataobj)
        imgs_per_mode.append(resize_data(np.transpose(full_brain, [-1, 0, 1])))

    # normalize each model then stack, stack after normalizing all modes together,
    # or stack without normalizing at all
    if normalization == 'by_mode':
        imgs = np.stack([normalize_stack(imgs) for imgs in imgs_per_mode], axis=-1)
    elif normalization == 'modes_together':
        imgs = normalize_stack(np.stack(imgs_per_mode, axis=-1))
    elif normalization is None:
        imgs = np.stack(imgs_per_mode, axis=-1)
    else:
        raise ValueError(f'{normalization} is not a supported normalization.')

    # get mask (labels)
    for file in need_files_oneof:
        if file in files:
            path = os.path.join(brain_path, file)
            full_brain_msk = np.array(nib.load(path).dataobj)
            msks = resize_data(parse_segments(full_brain_msk, msk_modes))
            break

    # determine which channels are wanted in our images in the case where we kept
    # extra scanning modes in order to normalize across all but only use a subset
    msk_mode_to_channel = {
        mode: channel_num for (channel_num, mode) in enumerate(msk_modes)
    }
    msk_channels_to_keep = np.array(
        [msk_mode_to_channel[mode] for mode in task_to_msk_modes[task]])
    # if we normalization is by_mode or None,
    # we have already restricted the image channels
    if normalization in ['by_mode', None]:
        img_channels_to_keep = np.arange(imgs.shape[-1])
    elif normalization == 'modes_together':
        img_mode_to_channel = {
            mode: channel_num for (channel_num, mode) in enumerate(img_modes)
        }
        img_channels_to_keep = np.array(
            [img_mode_to_channel[mode] for mode in task_to_img_modes[task]]
        )
    else:
        raise ValueError(f'{normalization} is not a supported normalization.')

    img = imgs
    msk = msks

    img, msk = _update_channels(
        img, msk, img_channels_to_keep, msk_channels_to_keep, channels_last)

    return img.astype(numpy_type), msk.astype(numpy_type)
