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
DATA_PATH_TEST = '../data/Apollo_16_Rev_17/'
ANNOTATIONS_PATH_TEST = '../data/Apollo_16_Rev_17/crater17_annotations.json'

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
    sample = train_dataset[i]

# Train the model
total_step = len(train_loader)
loss_list = []
acc_list = []

for epoch in range(num_epochs):
    for i, batch_instance in enumerate(train_loader):
        batch_instance, annotation_lengths, mask_status = batch_instance[0], batch_instance[1], batch_instance[2]
        images = torch.stack([t[0] for t in batch_instance])
        landmarks = torch.stack([t[1]['landmarks'] for t in batch_instance])
        print("epoch: ", epoch, "| Batch: ", i)

print("++++++++ Access Tests Passed ++++++++++")


