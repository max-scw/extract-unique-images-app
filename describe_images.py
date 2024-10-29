from pathlib import Path
from PIL import Image

import logging

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms


from typing import Union, List, Dict


torch.set_default_device("cpu")

def build_descriptor_model():
    # Load a pre-trained model
    model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    # Remove the classification layer (we only want the feature extractor)
    model = nn.Sequential(*list(model.children())[:-1])
    logging.info("Descriptor model built. Using a small MobileNetV3 pretrained on ImageNet.")
    # Set the model to evaluation mode
    model.eval()

    # set up preprocessing
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    return model, preprocess


def describe_image(img: Image, model, preprocess) -> torch.tensor:
    # preprocess image to apply the NN
    input_tensor = preprocess(img.convert("RGB"))
    input_batch = input_tensor.unsqueeze(0)  # Add a batch dimension
    # Encode the image using the pre-trained model
    with torch.no_grad():
        encoded_features = model(input_batch)
    return encoded_features.squeeze()


def calculate_feature_diffs(features1, features_all: list) -> List[float]:
    # calculate differences to already stored images
    return [float(((features1 - val) ** 2).mean().sqrt()) for val in features_all]
