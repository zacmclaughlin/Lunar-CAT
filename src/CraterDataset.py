from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import PyQt5
import matplotlib
matplotlib.use('qt5agg')
import matplotlib.pyplot as plt
import matplotlib.patches
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import json
import glob
from PIL import Image
import cv2
import random
import torchvision
from torchvision.transforms import functional as F
# from torch.utils.data.dataloader import default_collate


class CraterDataset(Dataset):
    """Crater dataset."""
    def __init__(self, root_dir, annotations_file, transform=None):
        """
        Args:
            annotations_file (string): Path to the json file with annotations.
            root_dir (string): Directory with all the images (has /JPGImages and /CraterMasks)
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.landmarks_frame = json.load(open(annotations_file))

        self.key_list = list(self.landmarks_frame.keys())
        for i in range(len(self.landmarks_frame)):
            if len(self.landmarks_frame[self.key_list[i]]['regions']) == 0:
                del self.landmarks_frame[self.key_list[i]]

        self.key_list = list(self.landmarks_frame.keys())
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.key_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        idx_keys = self.key_list[idx]
        image_name = os.path.join(self.root_dir + "JPGImages/",
                                  self.landmarks_frame[idx_keys]['filename'])

        image = io.imread(image_name)

        idx_annotations = []

        for i in range(len(self.landmarks_frame[idx_keys]['regions'])):
            cx = self.landmarks_frame[idx_keys]['regions'][i]['shape_attributes']['cx']
            cy = self.landmarks_frame[idx_keys]['regions'][i]['shape_attributes']['cy']
            r = self.landmarks_frame[idx_keys]['regions'][i]['shape_attributes']['r']
            idx_annotations.append([float(cx), float(cy), float(r)])

        sample = {'image': image, 'landmarks': np.array(idx_annotations)}

        if self.transform is not None:
            sample = self.transform(sample)

        image = sample['image']

        # create the black canvas
        segmented_image = np.zeros(shape=(image.shape[1], image.shape[2], image.shape[0]), dtype="uint8")

        for i in range(len(self.landmarks_frame[idx_keys]['regions'])):
            img = segmented_image
            center = (int((sample['landmarks'][i][0])), int((sample['landmarks'][i][1])))
            radius = int((sample['landmarks'][i][2]))
            color = (255 - (i + 1) * 2) % 255
            cv2.circle(img, center, radius, color, -1)

        # note that we haven't converted the mask to RGB,
        # because each color corresponds to a different instance
        # with 0 being background
        mask = cv2.cvtColor(segmented_image, cv2.COLOR_BGR2GRAY)

        segmented_image = Image.fromarray(segmented_image)
        segmented_image.save(self.root_dir + "CraterMasks/" +
                             self.landmarks_frame[idx_keys]['filename'].split(".")[0] +
                             "_mask.jpg")

        # convert the PIL Image into a numpy array
        mask = np.array(mask)
        # instances are encoded as different colors
        obj_ids = np.unique(mask)
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]

        # split the color-encoded mask into a set
        # of binary masks
        masks = mask == obj_ids[:, None, None]

        image_id = torch.tensor([idx])

        # get bounding box coordinates for each mask
        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)

        # there is only one class. and it is CRATER
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        # suppose all instances are not crowd
        is_crowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {"boxes": boxes,
                  "labels": labels,
                  "masks": masks,
                  "image_id": image_id,
                  "area": area,
                  "iscrowd": is_crowd}

        return image, target


class SquareCrop(object):
    """Crop a square form the sample (random orientation).

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                      left: left + new_w]

        landmarks[:, 0] = landmarks[:, 0] - left
        landmarks[:, 1] = landmarks[:, 1] - top
        landmarks[:, 2] = landmarks[:, 2]

        return {'image': image, 'landmarks': landmarks}


class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w))

        # h and w are swapped for landmarks because for images,
        # x and y axes are axis 1 and 0 respectively
        landmarks[:, 0] = landmarks[:, 0] * new_w / w
        landmarks[:, 1] = landmarks[:, 1] * new_h / h
        landmarks[:, 2] = landmarks[:, 2] * np.sqrt(new_w**2 + new_h**2) / np.sqrt(w**2 + h**2)

        return {'image': img, 'landmarks': landmarks}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image).float(),
                'landmarks': torch.from_numpy(landmarks)}


# def tag_crater(image, landmarks):
#     """Show image with evil craters"""
#     for i in range(len(landmarks)):
#         plt.imshow(image)
#         ax = plt.gca()
#         c = plt.Circle((landmarks[i, 0], landmarks[i, 1]), radius=landmarks[i, 2], fill=False, color='r')
#         ax.add_artist(c)
#         plt.pause(.001)


def collate_fn_crater_padding(batch):
    ''' Pads batch of variable length '''
    return tuple(zip(*batch))


def get_crater_image(root_dir='../data/Apollo_16_Rev_63/JPGImages/', image_name='random'):
    if image_name == 'random':
        image_name = random.choice(os.listdir(root_dir))
        print(image_name)
    image = Image.open(root_dir + image_name)
    image_tensor = F.to_tensor(image)
    return image_tensor
