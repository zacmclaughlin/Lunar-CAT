# importing torch utilities
import torchvision
import torch
from torch import utils
from torch.utils import data
from torchvision import transforms

# get models we care about
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

# import system hooks
import sys
from os import listdir
from os.path import isfile, join

# image data imports
from PIL import Image
from PyQt5.QtWidgets import QApplication
import numpy as np
import cv2

# local imports
import transforms as T
from engine import train_one_epoch, evaluate
import crater_dataset
from visualize_data import ImageView
import read_write_objects


# Pathways
DATA_PATH = '../data/Apollo_16_Rev_17/'
ANNOTATIONS_PATH = '../data/Apollo_16_Rev_17/crater17_annotations.json'
DATA_PATH_TEST = '../data/Apollo_16_Rev_28/'
ANNOTATIONS_PATH_TEST = '../data/Apollo_16_Rev_28/crater28_annotations.json'


def create_model_instance_segmentation(num_classes):
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


def load_model_instance_segmentation(num_classes, state_dict):
    # load an instance segmentation model pre-trained pre-trained on COCO
    loaded_model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

    # get number of input features for the classifier
    in_features = loaded_model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    loaded_model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = loaded_model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    loaded_model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)

    loaded_model.load_state_dict(state_dict)

    return loaded_model


def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


def create_model_output(model, path_to_images, model_filename):
    images = [f for f in listdir(path_to_images) if isfile(join(path_to_images, f))]
    image_tensors = []
    for image in range(int(len(images)/3)):
        image_tensors.append(crater_dataset.get_crater_image(root_dir=path_to_images,
                                                             image_name=images[image]))
        # pass a list of (potentially different sized) tensors
        # to the model, in 0-1 range. The model will take care of
        # batching them together and normalizing

    # get output dictionary
    output = model(image_tensors)

    output_network_test = '../output/' + model_filename + '.p'
    read_write_objects.save_obj_to_file(output_network_test, output)  # save to file

    return output


def train_and_evaluate(number_of_images):
    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    transform = transforms.Compose(
        [crater_dataset.Rescale(401), crater_dataset.SquareCrop(400), crater_dataset.ToTensor()])

    # our dataset has two classes only - background and person
    num_classes = 2
    # use our dataset and defined transformations
    dataset = crater_dataset.crater_dataset(DATA_PATH, ANNOTATIONS_PATH, transform)
    dataset_test = crater_dataset.crater_dataset(DATA_PATH_TEST, ANNOTATIONS_PATH_TEST, transform)

    # split the dataset in train and test set
    indices = torch.randperm(len(dataset)).tolist()
    dataset = torch.utils.data.Subset(dataset, indices[-number_of_images:])
    indices = torch.randperm(len(dataset_test)).tolist()
    dataset_test = torch.utils.data.Subset(dataset_test, indices[-number_of_images:])

    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=2, shuffle=True, num_workers=0,
        collate_fn=crater_dataset.collate_fn_crater_padding)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=0,
        collate_fn=crater_dataset.collate_fn_crater_padding)

    # get the model using our helper function
    model = create_model_instance_segmentation(num_classes)

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

    # let's train it for x epochs
    num_epochs = 10

    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        evaluate(model, data_loader_test, device=device)

    # confirm finish
    print("Finished training and evaluating")

    return model, dataset, dataset_test


def display_data(model, dataset):
    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # pick one image from the test set
    img, _ = dataset[0]
    # put the model in evaluation mode
    model.eval()
    with torch.no_grad():
        prediction = model([img.to(device)])

    a_crater = img.mul(255).permute(1, 2, 0).byte().numpy()
    a_guess_mask = Image.fromarray(prediction[0]['masks'][0, 0].mul(255).byte().cpu().numpy())
    bounding_boxes = np.asarray(prediction[0]['boxes'])

    for i in range(len(bounding_boxes)):  # range(len([0])):
        # print(np.asarray(prediction[0]['boxes'][i]))
        a_crater = cv2.rectangle(a_crater,
                      (bounding_boxes[i][0], bounding_boxes[i][1]),
                      (bounding_boxes[i][2], bounding_boxes[i][3]),
                      (0, 255, 0), 1)

    a_crater = Image.fromarray(a_crater)

    # show data
    imageshow = ImageView()
    imageshow.show_image([a_crater, a_guess_mask])
    imageshow.show()


def main():
    app = QApplication(sys.argv)

    model, training_data, evaluation_data = train_and_evaluate(number_of_images=20)

    create_model_output(model, '../data/Apollo_16_Rev_63/JPGImages/', 'bad_output')

    display_data(model=model, dataset=evaluation_data)

    torch.save(model.state_dict(), "../output/model.p")

    loaded_model = torch.load("../output/model.p")

    loaded_model = load_model_instance_segmentation(2, loaded_model)

    loaded_model.eval()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
