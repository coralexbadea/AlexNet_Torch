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
import random
from typing import Any
from torch import Tensor
from numpy import ndarray
import cv2
import numpy as np
import torch
from torchvision.transforms import functional as F_vision

__all__ = [
    "image_to_tensor", "tensor_to_image",
    "center_crop", "random_crop", "random_rotate", "random_vertically_flip", "random_horizontally_flip",
]


def image_to_tensor(image: np.ndarray, range_norm: bool, half: bool) -> torch.Tensor:
    """Convert the image data type to the Tensor (NCWH) data type supported by PyTorch

    Args:
        image (np.ndarray): The image data read by ``OpenCV.imread``, the data range is [0,255] or [0, 1]
        range_norm (bool): Scale [0, 1] data to between [-1, 1]
        half (bool): Whether to convert torch.float32 similarly to torch.half type

    Returns:
        tensor (torch.Tensor): Data types supported by PyTorch

    Examples:
        >>> example_image = cv2.imread("example_image.bmp")
        >>> example_tensor = image_to_tensor(example_image, False, False)

    """
    # Convert image data type to Tensor data type
    tensor = F_vision.to_tensor(image)

    # Scale the image data from [0, 1] to [-1, 1]
    if range_norm:
        tensor = tensor.mul(2.0).sub(1.0)## undefined - multiplies the element of the sensor by 2.0 and subtracts 1.0 from every element

    # Convert torch.float32 image data type to torch.half image data type
    if half:
        tensor = tensor.half()

    return tensor


def tensor_to_image(tensor: torch.Tensor, range_norm: bool, half: bool) -> Any:
    """Convert the Tensor(NCWH) data type supported by PyTorch to the np.ndarray(WHC) image data type

    Args:
        tensor (torch.Tensor): Data types supported by PyTorch (NCHW), the data range is [0, 1]
        range_norm (bool): Scale [-1, 1] data to between [0, 1]
        half (bool): Whether to convert torch.float32 similarly to torch.half type.

    Returns:
        image (np.ndarray): Data types supported by PIL or OpenCV

    Examples:
        >>> example_tensor = torch.randn([1,3, 256, 256], dtype=torch.float)
        >>> example_image = tensor_to_image(example_tensor, False, False)

    """
    # Scale the image data from [-1, 1] to [0, 1]
    if range_norm:
        tensor = tensor.add(1.0).div(2.0) ## undefined - adds 1.0 to every element and divides it by 2.0

    # Convert torch.float32 image data type to torch.half image data type
    if half:
        tensor = tensor.half()## undefined - converts the tensor from 32 bits to 16bits 
 
    image = tensor.squeeze(0).permute(1, 2, 0).mul(255).clamp(0, 255).cpu().numpy().astype("uint8")

    return image


def center_crop(
        images: ndarray | Tensor | list[ndarray] | list[Tensor],
        patch_size: int,
) -> [ndarray] or [Tensor] or [list[ndarray]] or [list[Tensor]]:
    if not isinstance(images, list):
        images = [images]

    # Detect input image data type
    input_type = "Tensor" if torch.is_tensor(images[0]) else "Numpy" ## undefined - if the input type is a tensor then it is set to Tensor, otherwise it is set to Numpy

    if input_type == "Tensor":
        image_height, image_width = images[0].size()[-2:]
    else:
        image_height, image_width = images[0].shape[0:2]

    # Calculate the start indices of the crop
    top = (image_height - patch_size) // 2 ## undefined - calculates the start indices for height for cropping by subtracting the patch size from the respective dimensions and dividing by 2
    left = (image_width - patch_size) // 2 ## undefined - calculates the start indices for width cropping by subtracting the patch size from the respective dimensions and dividing by 2

    # Crop lr image patch
    if input_type == "Tensor":
        images = [image[
                  :,
                  :,
                  top:top + patch_size,
                  left:left + patch_size] for image in images]
    else:
        images = [image[
                  top:top + patch_size,
                  left:left + patch_size,
                  ...] for image in images]

    # When image number is 1
    if len(images) == 1:
        images = images[0]

    return images


def random_crop(
        images: ndarray | Tensor | list[ndarray] | list[Tensor],
        patch_size: int,
) -> [ndarray] or [Tensor] or [list[ndarray]] or [list[Tensor]]:
    if not isinstance(images, list):
        images = [images]

    # Detect input image data type
    input_type = "Tensor" if torch.is_tensor(images[0]) else "Numpy" ## undefined - if the input type is tensor then it is set to tensor, otherwise to numpy

    if input_type == "Tensor":
        image_height, image_width = images[0].size()[-2:] ## undefined - gets the last two dimesnions of the tensor which correspond to height and witdh
    else:
        image_height, image_width = images[0].shape[0:2] ## undefined - gets the first two dimensions of the array which correspond to height and witdh

    # Just need to find the top and left coordinates of the image
    top = random.randint(0, image_height - patch_size) ## undefined -  randomly selects the top coordinates for cropping 
    left = random.randint(0, image_width - patch_size) ## undefined -  randomly selects the left coordinates for cropping 

    # Crop lr image patch
    if input_type == "Tensor":
        images = [image[
                  :,
                  :,
                  top:top + patch_size,
                  left:left + patch_size] for image in images]
    else:
        images = [image[
                  top:top + patch_size,
                  left:left + patch_size,
                  ...] for image in images]

    # When image number is 1
    if len(images) == 1:
        images = images[0]

    return images


def random_rotate( ## undefined - returns the rotated image
        images: ndarray | Tensor | list[ndarray] | list[Tensor],
        angles: list,
        center: tuple = None,
        rotate_scale_factor: float = 1.0
) -> [ndarray] or [Tensor] or [list[ndarray]] or [list[Tensor]]:
    # Random select specific angle
    angle = random.choice(angles) ## undefined - random selects an angle

    if not isinstance(images, list): ## undefined - if the input isn't a list
        images = [images] ## undefined - converts it

    # Detect input image data type
    input_type = "Tensor" if torch.is_tensor(images[0]) else "Numpy" ## undefined - determines the input type bu checking the input type

    if input_type == "Tensor":
        image_height, image_width = images[0].size()[-2:] ## undefined - the height and width are the last two for tensor
    else:
        image_height, image_width = images[0].shape[0:2] ## undefined - first two for numpy

    # Rotate LR image
    if center is None:
        center = (image_width // 2, image_height // 2) ## undefined - calculates the center

    matrix = cv2.getRotationMatrix2D(center, angle, rotate_scale_factor)## undefined - converts the rotation matrix

    if input_type == "Tensor":
        images = [F_vision.rotate(image, angle, center=center) for image in images]## undefined - rotates the image if the input is of tensor type
    else:
        images = [cv2.warpAffine(image, matrix, (image_width, image_height)) for image in images]## undefined - rotates the image if the input is of numpy type

    # When image number is 1
    if len(images) == 1:## undefined - result is returned as one image if the input is one image
        images = images[0]## undefined - returns a list of images

    return images


def random_horizontally_flip(
        images: ndarray | Tensor | list[ndarray] | list[Tensor],
        p: float = 0.5
) -> [ndarray] or [Tensor] or [list[ndarray]] or [list[Tensor]]:
    # Get horizontal flip probability
    flip_prob = random.random()

    if not isinstance(images, list):
        images = [images]

    # Detect input image data type
    input_type = "Tensor" if torch.is_tensor(images[0]) else "Numpy"

    if flip_prob > p:
        if input_type == "Tensor":
            images = [F_vision.hflip(image) for image in images]
        else:
            images = [cv2.flip(image, 1) for image in images]

    # When image number is 1
    if len(images) == 1:
        images = images[0]

    return images


def random_vertically_flip(
        images: ndarray | Tensor | list[ndarray] | list[Tensor],
        p: float = 0.5
) -> [ndarray] or [Tensor] or [list[ndarray]] or [list[Tensor]]:
    # Get vertical flip probability
    flip_prob = random.random()

    if not isinstance(images, list):
        images = [images]

    # Detect input image data type
    input_type = "Tensor" if torch.is_tensor(images[0]) else "Numpy"

    if flip_prob > p:
        if input_type == "Tensor":
            images = [F_vision.vflip(image) for image in images]
        else:
            images = [cv2.flip(image, 0) for image in images]

    # When image number is 1
    if len(images) == 1:
        images = images[0]

    return images
