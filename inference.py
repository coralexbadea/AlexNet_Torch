# Copyright 2022 Dakewe Biotech Corporation. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import argparse
import json
import os

import cv2
import torch
from PIL import Image
from torch import nn
from torchvision.transforms import Resize, ConvertImageDtype, Normalize ## undefined - imports the classes Resize, ConvertImageDtype, Normalize

import config
import imgproc
import model
from utils import load_state_dict

model_names = sorted( ## undefined - contains a sorted list of model names
    name for name in model.__dict__ if name.islower() and not name.startswith("__") and callable(model.__dict__[name]))


def load_class_label(class_label_file: str, num_classes: int) -> list:
    class_label = json.load(open(class_label_file))
    class_label_list = [class_label[str(i)] for i in range(num_classes)]

    return class_label_list


def choice_device(device_type: str) -> torch.device:
    # Select model processing equipment type
    if device_type == "cuda":
        device = torch.device("cuda", 0)
    else:
        device = torch.device("cpu")
    return device


def build_model(model_arch_name: str, model_num_classes: int, device: torch.device) -> [nn.Module, nn.Module]: ## undefined - build_model function that takes as param a string, int and torch.device
    alexnet_model = model.__dict__[model_arch_name](num_classes=model_num_classes) ## undefined - creates an instance of the model architecture
    alexnet_model = alexnet_model.to(device=device, memory_format=torch.channels_last) ## undefined - moves the model to the target device and specifies the memory format

    return alexnet_model


def preprocess_image(image_path: str, image_size: int, device: torch.device) -> torch.Tensor: ## undefined - preprocess method 
    image = cv2.imread(image_path) ## undefined - reads the image from the given path

    # BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)## undefined - converts the image from BGR to RGB

    # OpenCV convert PIL
    image = Image.fromarray(image) ## undefined - converts the image to a PIL format from a numpy array format

    # Resize to 224
    image = Resize([image_size, image_size])(image) ## undefined - resizes the image 
    # Convert image data to pytorch format data
    tensor = imgproc.image_to_tensor(image, False, False).unsqueeze_(0) ## undefined -  converts the PIL Image to a tensor
    # Convert a tensor image to the given ``dtype`` and scale the values accordingly
    tensor = ConvertImageDtype(torch.float)(tensor) ## undefined -  converts the data type of the tensor image to torch.float
    # Normalize a tensor image with mean and standard deviation.
    tensor = Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(tensor) ## undefined -  normalizes the tensor image using the mean and standard deviation values 

    # Transfer tensor channel image format data to CUDA device
    tensor = tensor.to(device=device, memory_format=torch.channels_last, non_blocking=True) ## undefined -  moves the tensor to the target device

    return tensor


def main(): ## undefined - main function
    # Get the label name corresponding to the drawing
    class_label_map = load_class_label(args.class_label_file, args.model_num_classes) ## undefined - loads the class label mapping

    device = choice_device(args.device_type)## undefined - selects the target device

    # Initialize the model
    alexnet_model = build_model(args.model_arch_name, args.model_num_classes, device) ## undefined - builds the model specified by the parameters
    print(f"Build {config.model_arch_name.upper()} model successfully.")

    # Load model weights
    alexnet_model, _, _, _, _, _ = load_state_dict(alexnet_model, args.model_weights_path)## undefined - loads the pretrained weights for the model
    print(f"Load {config.model_arch_name.upper()} "
          f"model weights `{os.path.abspath(args.model_weights_path)}` successfully.")

    # Start the verification mode of the model.
    alexnet_model.eval()## undefined - sets the model in evaluation mode

    tensor = preprocess_image(args.image_path, args.image_size, device)## undefined - preprocesses the input image specified

    # Inference
    with torch.no_grad():## undefined - creates a context where no gradients are computed for efficiency
        output = alexnet_model(tensor)## undefined - performs inference on the preprocessed input image by passing it through the alexnet_model

    # Calculate the five categories with the highest classification probability
    prediction_class_index = torch.topk(output, k=5).indices.squeeze(0).tolist()## undefined - etrieves the top 5 class indices with the highest classification probabilities

    # Print classification results
    for class_index in prediction_class_index: ## undefined - iterates through the prediction_class_index
        prediction_class_label = class_label_map[class_index] ## undefined - etrieves the corresponding class label for the current class index
        prediction_class_prob = torch.softmax(output, dim=1)[0, class_index].item() ## undefined - calculates the classification probability for the current class index
        print(f"{prediction_class_label:<75} ({prediction_class_prob * 100:.2f}%)") ## undefined - prints the class label and its corresponding probability


if __name__ == "__main__": ## undefined - entry point of the script
    parser = argparse.ArgumentParser() ## undefined - sets up an argument parser
    parser.add_argument("--model_arch_name", type=str, default="alexnet") ## undefined - adds the argument for the model 
    parser.add_argument("--class_label_file", type=str, default="./data/ImageNet_1K_labels_map.txt") ## undefined - adds the argument for the class label
    parser.add_argument("--model_num_classes", type=int, default=1000) ## undefined - adds the argument for the number of model classes
    parser.add_argument("--model_weights_path", type=str,
                        default="./results/pretrained_models/AlexNet-ImageNet_1K-9df8cd0f.pth.tar") ## undefined - adds the argument for the nr of weights plan
    parser.add_argument("--image_path", type=str, default="./figure/n01440764_36.JPEG")
    parser.add_argument("--image_size", type=int, default=224) ## undefined - adds the argument for the size of the image
    parser.add_argument("--device_type", type=str, default="cpu", choices=["cpu", "cuda"]) ## undefined - adds the argument for the device type
    args = parser.parse_args()

    main()
