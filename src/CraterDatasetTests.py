import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets
from bokeh.plotting import figure
from bokeh.io import show
from bokeh.models import LinearAxis, Range1d
import numpy as np
import crater_dataset


print("++++++++ Begin Access Tests ++++++++++")

# Hyperparameters
num_epochs = 1
num_classes = 10
batch_size = 4
learning_rate = 0.001

MODEL_STORE_PATH = '../data/'

DATA_PATH = '../data/Apollo_16_Rev_17/'
ANNOTATIONS_PATH = '../data/Apollo_16_Rev_17/crater17_annotations.json'
DATA_PATH_TEST = '../data/Apollo_16_Rev_28/'
ANNOTATIONS_PATH_TEST = '../data/Apollo_16_Rev_28/crater28_annotations.json'

from albumentations import (
    PadIfNeeded,
    HorizontalFlip,
    VerticalFlip,
    CenterCrop,
    Crop,
    Compose,
    Transpose,
    RandomRotate90,
    ElasticTransform,
    GridDistortion,
    OpticalDistortion,
    RandomSizedCrop,
    OneOf,
    CLAHE,
    RandomBrightnessContrast,
    RandomGamma
)

transform = transforms.Compose([crater_dataset.Rescale(401), crater_dataset.SquareCrop(400), crater_dataset.ToTensor()])

# zacs evil dataset...
train_dataset = crater_dataset.crater_dataset(DATA_PATH, ANNOTATIONS_PATH, transform)
test_dataset = crater_dataset.crater_dataset(DATA_PATH_TEST, ANNOTATIONS_PATH_TEST, transform)

# Data [redacted]
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size,
                          shuffle=True, collate_fn=crater_dataset.collate_fn_crater_padding)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size,
                         shuffle=False, collate_fn=crater_dataset.collate_fn_crater_padding)

for i in range(len(train_dataset)):
    aug = PadIfNeeded(p=1, min_height=400, min_width=400)
    augmented = aug(image=train_dataset[i][0].numpy(),
                    mask=train_dataset[i][1]['masks'].numpy(),
                    bbox=train_dataset[i][1]['boxes'])

    print(type(train_dataset[i][0]))
    # train_dataset[i][0] = augmented['image']
    # train_dataset[i][1]['mask'] = augmented['mask']
    # train_dataset[i][1]['boxes'] = augmented['bbox']


print("++++++++ Access Tests Passed ++++++++++")


