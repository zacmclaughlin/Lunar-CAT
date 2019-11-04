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
import datetime

# image data imports
from PIL import Image
from PyQt5.QtWidgets import QApplication
import numpy as np
import cv2

# data augmentation imports
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

# local imports
import transforms as T
from engine import train_one_epoch, evaluate
import crater_dataset
from visualize_data import ImageView, ImageBook
import read_write_objects


# Pathways
DATA_PATH = '../data/Apollo_16_Rev_18/'
ANNOTATIONS_PATH = '../data/Apollo_16_Rev_18/crater18_annotations.json'

DATA_PATH_TEST = '../data/Apollo_16_Rev_28/'
ANNOTATIONS_PATH_TEST = '../data/Apollo_16_Rev_28/crater28_annotations.json'

LOAD_MODEL_FILE_AND_PATH = "../output/nov3_overnight_run_60images_20epochs.p"
LOAD_OUTPUT_FILE_AND_PATH = ""

currentDT = datetime.datetime.now()
currentDT = str(currentDT).replace(":", "-").replace(" ", "--").split(".")[0]

SAVE_MODEL_FILE_AND_PATH = "../output/model_at_time_" + currentDT + ".p"
SAVE_OUTPUT_FILE_AND_PATH = "../output/output_at_time_" + currentDT + ".p"

# Other dataset paths:
# '../data/Apollo_16_Rev_17/'
# '../data/Apollo_16_Rev_17/crater17_annotations.json'
# '../data/Apollo_16_Rev_18/'
# '../data/Apollo_16_Rev_18/crater18_annotations.json'
# '../data/Apollo_16_Rev_28/'
# '../data/Apollo_16_Rev_28/crater28_annotations.json'
# '../data/Apollo_16_Rev_63/'
# '../data/Apollo_16_Rev_63/crater63_annotations.json'


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
    loaded_model = create_model_instance_segmentation(num_classes)
    loaded_model.load_state_dict(state_dict)
    return loaded_model


def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


def create_model_output(model, path_to_images, model_filename=SAVE_OUTPUT_FILE_AND_PATH):
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
    read_write_objects.save_obj_to_file(model_filename, output)  # save to file

    return output


def get_crater_datasets(number_of_images):

    transform = transforms.Compose(
        [crater_dataset.Rescale(401), crater_dataset.SquareCrop(400), crater_dataset.ToTensor()])

    aug = Compose([PadIfNeeded(p=1, min_height=400, min_width=400),
                   # VerticalFlip(p=0.65),
                   # RandomRotate90(p=0.7),
                   # GridDistortion(p=1),
                   Transpose(p=1),
                   OpticalDistortion(p=1, distort_limit=.5, shift_limit=0.2)
                   ])

    # use our dataset and defined transformations
    dataset = crater_dataset.crater_dataset(DATA_PATH, ANNOTATIONS_PATH, transform, augmentation=aug)
    dataset_test = crater_dataset.crater_dataset(DATA_PATH_TEST, ANNOTATIONS_PATH_TEST, transform, augmentation=aug)

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

    return dataset, data_loader, dataset_test, data_loader_test


def train_and_evaluate(model, data_loader, data_loader_test, num_epochs):
    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

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
    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        evaluate(model, data_loader_test, device=device)
        # intermediate saving
        torch.save(model.state_dict(), SAVE_MODEL_FILE_AND_PATH)

    # confirm finish
    print("Finished training and evaluating")

    return model


def get_display_widget(model, dataset):
    image_set = {}
    for datum in range(len(dataset)):
        # train on the GPU or on the CPU, if a GPU is not available
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        # pick one image from the test set
        img, target = dataset[datum]
        # put the model in evaluation mode
        model.eval()
        with torch.no_grad():
            prediction = model([img.to(device)])

        this_crater = img.mul(255).permute(1, 2, 0).byte().numpy()
        guess_mask = np.zeros((prediction[0]['masks'].shape[2], prediction[0]['masks'].shape[3]))
        bounding_boxes = np.asarray(prediction[0]['boxes'])
        for j in range(int(prediction[0]['masks'].shape[0]/10)):
            mask = prediction[0]['masks'][j, 0].mul(255).byte().cpu().numpy()
            guess_mask = guess_mask + mask        
            this_crater = cv2.rectangle(this_crater,
                         (bounding_boxes[j][0], bounding_boxes[j][1]),
                         (bounding_boxes[j][2], bounding_boxes[j][3]),
                         (0, 255, 0), 1)

        this_guess_mask = Image.fromarray(guess_mask)
        this_crater = Image.fromarray(this_crater)
        image_canvas = ImageView()
        image_canvas.set_image([this_crater, this_guess_mask], target)
        image_set[str(datum)] = image_canvas

    return ImageBook(image_set)  # return display widget


def get_run_type(arguments, number_of_arguments):
    if number_of_arguments > 1:
        position = 1
        while number_of_arguments >= position:
            print("parameter %i: %s" % (position, sys.argv[position]))
            position = position + 1
        print("At this time, only one command line argument at a time is supported. \n "
              "Please choose from the following options: \n"
              "-viz, -new, -load \n"
              "-new will be assumed by default. ")
        return "-new"
    elif number_of_arguments == 0:
        return "-new"
    else:
        return arguments[1]


def main(arguments):

    run_type = get_run_type(arguments, len(arguments) - 1)

    dataset, data_loader, dataset_test, data_loader_test = get_crater_datasets(number_of_images=2)

    if run_type == "-load":
        # Load the model
        loaded_model = torch.load(LOAD_MODEL_FILE_AND_PATH)
        loaded_model = load_model_instance_segmentation(2, loaded_model)

        # Train the model
        model = train_and_evaluate(loaded_model, data_loader, data_loader_test, num_epochs=1)

        # Save the model
        torch.save(model.state_dict(), SAVE_MODEL_FILE_AND_PATH)
        create_model_output(model, DATA_PATH_TEST, SAVE_OUTPUT_FILE_AND_PATH)

    elif run_type == "-viz":
        # Load the model
        loaded_model = torch.load(LOAD_MODEL_FILE_AND_PATH)
        loaded_model = load_model_instance_segmentation(2, loaded_model)

        # Visualize the model
        app = QApplication(sys.argv)
        image_book = get_display_widget(model=loaded_model, dataset=dataset_test)
        image_book.show()
        sys.exit(app.exec_())

    elif run_type == "-new":
        # Create a new model using our helper function. Our dataset has two classes only - background and crater
        model = create_model_instance_segmentation(2)

        # Train the model
        model = train_and_evaluate(model, data_loader, data_loader_test, num_epochs=10)

        # Save the model
        torch.save(model.state_dict(), SAVE_MODEL_FILE_AND_PATH)
        create_model_output(model, DATA_PATH_TEST, SAVE_OUTPUT_FILE_AND_PATH)

    else:
        print("Please choose from the following options: \n"
              "-viz, -new, -load \n"
              "-new will be assumed if no flags are passed.")


if __name__ == "__main__":
    main(sys.argv)
