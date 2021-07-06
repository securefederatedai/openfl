# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Shard descriptor."""

import os
import xml.etree.ElementTree as ElementTree

import numpy as np
from PIL import Image


class ShardDescriptor:
    """Shard descriptor class."""

    def __len__(self):
        """Return the len of the dataset."""
        raise NotImplementedError

    def __getitem__(self, index: int):
        """Return a item by the index."""
        raise NotImplementedError

    @property
    def sample_shape(self):
        """Return the sample shape info."""
        raise NotImplementedError

    @property
    def target_shape(self):
        """Return the target shape info."""
        raise NotImplementedError

    @property
    def dataset_description(self) -> str:
        """Return the dataset description."""
        return ''


class VOCDatasetSD(ShardDescriptor):
    """VOCDataset shard descriptor class."""

    class_names = ('__background__',
                   'aeroplane', 'bicycle', 'bird', 'boat',
                   'bottle', 'bus', 'car', 'cat', 'chair',
                   'cow', 'diningtable', 'dog', 'horse',
                   'motorbike', 'person', 'pottedplant',
                   'sheep', 'sofa', 'train', 'tvmonitor')

    def __init__(self, data_dir, split, keep_difficult=False):
        """Dataset for VOC data.
        Args:
            data_dir: the root of the VOC2007 or VOC2012 dataset,
                the directory contains the following sub-directories:
                Annotations, ImageSets, JPEGImages, SegmentationClass, SegmentationObject.
        """
        self.data_dir = data_dir
        self.split = split
        image_sets_file = os.path.join(self.data_dir, 'ImageSets', 'Main', f'{self.split}.txt')
        self.ids = self.__class__._read_image_ids(image_sets_file)
        self.keep_difficult = keep_difficult

        self.class_dict = {class_name: i for i, class_name in enumerate(self.class_names)}

    def __getitem__(self, index: int):
        """Return a item by the index."""
        img = self._read_image(index)
        target = self.get_annotation(index)[1]
        return (img, target)

    def __len__(self):
        """Return the len of the dataset."""
        return len(self.ids)

    @property
    def sample_shape(self):
        """Return the sample shape info."""
        return self.get_img_info(0)

    @property
    def target_shape(self) -> int:
        """Return the target shape info."""
        raise NotImplementedError

    @property
    def dataset_description(self) -> str:
        """Return the dataset description."""
        return 'VOC2007' if '2007' in self.data_dir else 'VOC2012'

    def get_annotation(self, index):
        """Return an annotation."""
        image_id = self.ids[index]
        return image_id, self._get_annotation(image_id)

    @staticmethod
    def _read_image_ids(image_sets_file):
        """Read image ids."""
        ids = []
        with open(image_sets_file) as f:
            for line in f:
                ids.append(line.rstrip())
        return ids

    def _get_annotation(self, image_id):
        """Return an annotation."""
        annotation_file = os.path.join(self.data_dir, 'Annotations', f'{image_id}.xml')
        objects = ElementTree.parse(annotation_file).findall('object')
        boxes = []
        labels = []
        is_difficult = []
        for obj in objects:
            class_name = obj.find('name').text.lower().strip()
            bbox = obj.find('bndbox')
            # VOC dataset format follows Matlab, in which indexes start from 0
            x1 = float(bbox.find('xmin').text) - 1
            y1 = float(bbox.find('ymin').text) - 1
            x2 = float(bbox.find('xmax').text) - 1
            y2 = float(bbox.find('ymax').text) - 1
            boxes.append([x1, y1, x2, y2])
            labels.append(self.class_dict[class_name])
            is_difficult_str = obj.find('difficult').text
            is_difficult.append(int(is_difficult_str) if is_difficult_str else 0)

        return (np.array(boxes, dtype=np.float32),
                np.array(labels, dtype=np.int64),
                np.array(is_difficult, dtype=np.uint8))

    def get_img_info(self, index):
        """Return an image info."""
        img_id = self.ids[index]
        annotation_file = os.path.join(self.data_dir, 'Annotations', f'{img_id}.xml')
        anno = ElementTree.parse(annotation_file).getroot()
        size = anno.find('size')
        im_info = tuple(map(int, (size.find('height').text, size.find('width').text)))
        return np.array[im_info[0], im_info[1], 3]

    def _read_image(self, image_id):
        """Read image."""
        image_file = os.path.join(self.data_dir, 'JPEGImages', f'{image_id}.jpg')
        image = Image.open(image_file).convert('RGB')
        image = np.array(image)
        return image
