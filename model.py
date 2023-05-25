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
from typing import Any

import torch ## undefined - imports the torch module
from torch import Tensor ## undefined - imports Tensor class from torch
from torch import nn ## undefined - imports the nn function from torch

__all__ = [
    "AlexNet",
    "alexnet",
]


class AlexNet(nn.Module): ## undefined - sublcass of nn.Module
    def __init__(self, num_classes: int = 1000) -> None: ## undefined - initializes the number of output classes
        super(AlexNet, self).__init__() ## undefined - it calls the method of the parent class

        self.features = nn.Sequential( ## undefined - defines the self.features attributes
            nn.Conv2d(3, 64, (11, 11), (4, 4), (2, 2)), ## undefined - creates a 2D convolutional layer with 3 input channels, 64 output channels, an 11x11 kernel size, a stride of 4 in both dimensions, and a padding of 2
            nn.ReLU(True), ## undefined - ReLU activation function
            nn.MaxPool2d((3, 3), (2, 2)), ## undefined - performs 2D max pooling with a pool size of 3x3 and a stride of 2. It reduces the spatial dimensions of the input

            nn.Conv2d(64, 192, (5, 5), (1, 1), (2, 2)), ## undefined - creates another 2D convolutional layer with 64 input channels, 192 output channels, a 5x5 kernel size, a stride of 1, and a padding of 2
            
            nn.ReLU(True), ## undefined - ReLU activation function
            nn.MaxPool2d((3, 3), (2, 2)), ## undefined - layer reduces the spatial dimensions further

            nn.Conv2d(192, 384, (3, 3), (1, 1), (1, 1)), ## undefined - creates another 2D convolutional layer with 192 input channels, 384 output channels, a 3x3 kernel size, a stride of 1, and a padding of 1
            nn.ReLU(True),
            nn.Conv2d(384, 256, (3, 3), (1, 1), (1, 1)), ## undefined - creates another 2D convolutional layer with 384 input channels, 256 output channels, a 3x3 kernel size, a stride of 1, and a padding of 1
            nn.ReLU(True), 
            nn.Conv2d(256, 256, (3, 3), (1, 1), (1, 1)), ## undefined - creates another 2D convolutional layer with 256 input channels, 256 output channels, a 3x3 kernel size, a stride of 1, and a padding of 1
            nn.ReLU(True),
            nn.MaxPool2d((3, 3), (2, 2)), ## undefined - layer reduces the spatial dimensions further
        )

        self.avgpool = nn.AdaptiveAvgPool2d((6, 6)) ## undefined - creates an adaptive average pooling layer

        self.classifier = nn.Sequential( ## undefined - sequential container that defines the classification part of the AlexNet model
            nn.Dropout(0.5), ## undefined - applies dropout regularization with a probability of 0.5
            nn.Linear(256 * 6 * 6, 4096), ## undefined - takes the flattened feature map from the previous layer and produces a 4096-dimensional output
            nn.ReLU(True), 
            nn.Dropout(0.5), ## undefined -  layer is applied
            nn.Linear(4096, 4096), ## undefined - fully connected layer is added
            nn.ReLU(True),
            nn.Linear(4096, num_classes), ## undefined - produces the output logits for classification
        )

    def forward(self, x: Tensor) -> Tensor: ## undefined - takes and returns a Tensor 
        return self._forward_impl(x) ## undefined - performs the forward propagation through the network layers

    # Support torch.script function
    def _forward_impl(self, x: Tensor) -> Tensor: ## undefined - takes and returns a Tensor 
        out = self.features(x) ## undefined -  consists of the convolutional and pooling layers
        out = self.avgpool(out) ## undefined - erforms adaptive average pooling to produce a fixed-size representation
        out = torch.flatten(out, 1) ## undefined - the output tensor is flattened along the second dimension
        out = self.classifier(out) ## undefined - consists of fully connected layers for classification

        return out


def alexnet(**kwargs: Any) -> AlexNet: ## undefined - takes any parameters and returns AlexNet type
    model = AlexNet(**kwargs) ## undefined -  creates an instance of the AlexNet class, passing any keyword arguments kwargs to the constructor

    return model ## undefined - returns the created model
