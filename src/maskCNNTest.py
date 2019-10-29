import torchvision
import torch
from torch import utils
from torch.utils import data
from torchvision import transforms
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.faster_rcnn import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import transforms as T
from engine import train_one_epoch, evaluate
import utils
import CraterDataset


def get_model_instance_segmentation(num_classes):
    # load an instance segmentation model pre-trained pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)

    return model


def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


def main():
    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    DATA_PATH = '../data/Apollo_16_Rev_17/'
    ANNOTATIONS_PATH = '../data/Apollo_16_Rev_17/crater17_annotations.json'
    DATA_PATH_TEST = '../data/Apollo_16_Rev_28/'
    ANNOTATIONS_PATH_TEST = '../data/Apollo_16_Rev_28/crater28_annotations.json'

    transform = transforms.Compose(
        [CraterDataset.Rescale(401), CraterDataset.SquareCrop(400), CraterDataset.ToTensor()])

    # zacs dataset
    train_dataset = CraterDataset.CraterDataset(DATA_PATH, ANNOTATIONS_PATH, transform)
    test_dataset = CraterDataset.CraterDataset(DATA_PATH_TEST, ANNOTATIONS_PATH_TEST, transform)

    # our dataset has two classes only - background and person
    num_classes = 2
    # use our dataset and defined transformations
    dataset = CraterDataset.CraterDataset(DATA_PATH, ANNOTATIONS_PATH, transform)
    dataset_test = CraterDataset.CraterDataset(DATA_PATH_TEST, ANNOTATIONS_PATH_TEST, transform)

    # dataset = CraterDataset.CraterDataset('PennFudanPed', get_transform(train=True))
    # dataset_test = CraterDataset.CraterDataset('PennFudanPed', get_transform(train=False))

    # split the dataset in train and test set
    # indices = torch.randperm(len(dataset)).tolist()
    # dataset = torch.utils.data.Subset(dataset, indices[:-30])
    # dataset_test = torch.utils.data.Subset(dataset_test, indices[-30:])

    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=2, shuffle=True, num_workers=0,
        collate_fn=CraterDataset.collate_fn_crater_padding)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=0,
        collate_fn=CraterDataset.collate_fn_crater_padding)

    # get the model using our helper function
    model = get_model_instance_segmentation(num_classes)

    # move model to the right device
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)

    # let's train it for 10 epochs
    num_epochs = 10

    for epoch in range(num_epochs):
        print("Epoch: ", epoch)
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10) # this worked
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        evaluate(model, data_loader_test, device=device)
        print("end epoch", epoch)

    print("That's it!")


if __name__ == "__main__":
    main()
