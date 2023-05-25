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
import queue ## undefined - imports the queue module
import sys ## undefined - imports the sys module
import threading ## undefined - imports the threading module
from glob import glob ## undefined - imports the glob function from the glob module

import cv2 ## undefined - imports the cv2 module
import torch ## undefined - imports the torch module
from PIL import Image ## undefined - imports the Image class from the PIL module
from torch.utils.data import Dataset, DataLoader ## undefined - imports the classes Dataset, DataLoader from the torch.utils.data module
from torchvision import transforms ## undefined - imports the transforms module from the torchvision package
from torchvision.datasets.folder import find_classes ## undefined - imports find_classes function from torchvision.datasets.folder
from torchvision.transforms import TrivialAugmentWide ## undefined - imports TrivialAugmentWide class from torchvision.transforms

import imgproc

__all__ = [
    "ImageDataset",
    "PrefetchGenerator", "PrefetchDataLoader", "CPUPrefetcher", "CUDAPrefetcher",
]

# Image formats supported by the image processing library
IMG_EXTENSIONS = ("jpg", "jpeg", "png", "ppm", "bmp", "pgm", "tif", "tiff", "webp")

# The delimiter is not the same between different platforms
if sys.platform == "win32": ## undefined - if the platfrom is win32
    delimiter = "\\" ## undefined this is the delimiter
else: ## undefined if the platform isnt that one
    delimiter = "/" ## undefined - this is the delimiter

 
class ImageDataset(Dataset): ## undefined  this class is a subclass of Dataset
    """Define training/valid dataset loading methods.

    Args:
        image_dir (str): Train/Valid dataset address.
        image_size (int): Image size.
        mode (str): Data set loading method, the training data set is for data enhancement,
            and the verification data set is not for data enhancement.
    """

    def __init__(self, image_dir: str, image_size: int, mode: str) -> None:
        super(ImageDataset, self).__init__() ## undefined -  calls the constructor of the parent class ImageDataset
        # Iterate over all image paths
        self.image_file_paths = glob(f"{image_dir}/*/*")
        # Form image class label pairs by the folder where the image is located
        _, self.class_to_idx = find_classes(image_dir)
        self.image_size = image_size ## undefined - assigns the image size 
        self.mode = mode ## undefined - assigns the mode in which the datasheet is being used, train, valid, test
        self.delimiter = delimiter## undefined - assigns the type of delimiter

        if self.mode == "Train": ## undefined - if the mode that was assigned is "train"
            # Use PyTorch's own data enhancement to enlarge and enhance data
            self.pre_transform = transforms.Compose([ ## undefined - creates a transformation pipeline consisting in multiple image transformations
                transforms.RandomResizedCrop(self.image_size), ## undefined - performs a random crop of the input image and then resizes it
                TrivialAugmentWide(), ## undefined - 
                transforms.RandomRotation([0, 270]), ## undefined -  randomly rotates the input image by an angle between 0 and 270 degrees
                transforms.RandomHorizontalFlip(0.5), ## undefined -  horizontally flips the input image with a probability of 0.5
                transforms.RandomVerticalFlip(0.5), ## undefined - vertically flips the input image with a probability of 0.5
            ])
        elif self.mode == "Valid" or self.mode == "Test": ## undefined if the mode is either valid or test
            # Use PyTorch's own data enhancement to enlarge and enhance data
            self.pre_transform = transforms.Compose([ ## undefined - creates a transformation pipeline
                transforms.Resize(256), ## undefined - resizing the image 
                transforms.CenterCrop([self.image_size, self.image_size]), ## undefined - performs a center crop
            ])
        else:
            raise "Unsupported data read type. Please use `Train` or `Valid` or `Test`"

        self.post_transform = transforms.Compose([ ## undefined - creates a composition of multiple image transformations
            transforms.ConvertImageDtype(torch.float), ## undefined - converts the image data type to torch.float
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) ## undefined -  normalizes the image tensor by subtracting the mean and dividing by the standard deviation
        ])

    def __getitem__(self, batch_index: int) -> [torch.Tensor, int]: ## undefined - defines the method and the expected return type
        image_dir, image_name = self.image_file_paths[batch_index].split(self.delimiter)[-2:]
        # Read a batch of image data
        if image_name.split(".")[-1].lower() in IMG_EXTENSIONS: ## undefined - checks if the file extension of the image is supported, it splits the image_name by"." to extract the file extension, the [-1] indexing selects the last element, which represents the file extension, the lower() method converts the file extension to lowercase
            image = cv2.imread(self.image_file_paths[batch_index]) ## undefined -  reads the image data using OpenCV's imread function
            target = self.class_to_idx[image_dir] ## undefined - assigns a target label to the image based on its directory
        else: ## undefined - if the file extension isn't supported
            raise ValueError(f"Unsupported image extensions, Only support `{IMG_EXTENSIONS}`, "
                             "please check the image file extensions.")

        # BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) ## undefined - converts the image from BGR to RGB

        # OpenCV convert PIL
        image = Image.fromarray(image) ## undefined -  converts the image from a NumPy array format (using OpenCV) to a PIL (Python Imaging Library) image format

        # Data preprocess
        image = self.pre_transform(image) ## undefined - he image is processed through the transformation pipeline defined by pre_transform

        # Convert image data into Tensor stream format (PyTorch).
        # Note: The range of input and output is between [0, 1]
        tensor = imgproc.image_to_tensor(image, False, False) ## undefined - converts the processed PIL image into a tensor format compatible with PyTorch

        # Data postprocess
        tensor = self.post_transform(tensor) ## undefined - series of transformations after the processing 

        return {"image": tensor, "target": target}

    def __len__(self) -> int:
        return len(self.image_file_paths) ## undefined - returns the length of image_file_path list


class PrefetchGenerator(threading.Thread): ## The PrefetchGenerator class is a subclass of threading.Thread
    """A fast data prefetch generator.

    Args:
        generator: Data generator.
        num_data_prefetch_queue (int): How many early data load queues.
    """

    def __init__(self, generator, num_data_prefetch_queue: int) -> None:
        threading.Thread.__init__(self)
        self.queue = queue.Queue(num_data_prefetch_queue) ## undefined - creates a queue with a max size of num_data_prefetch_queue
        self.generator = generator ## undefined - storing the data generator for later use
        self.daemon = True ## undefined - a daemon thread is a background thread that doesn't prevent the program from exiting if it's still running
        self.start() ## undefined - starts the thread

    def run(self) -> None: ## undefined - this method isn't expecting a return value
        for item in self.generator: ## undefined - iterates through generator
            self.queue.put(item) ## undefined - puts each item in the queue
        self.queue.put(None) ## undefined - puts none as a signal that there aren't any other items

    def __next__(self): ## undefined - allows instances of the class to be used as an iterator
        next_item = self.queue.get() ## undefined - retrieves the next item
        if next_item is None: ## undefined - if there isn't a next item
            raise StopIteration ## undefined - it stops
        return next_item ## undefined - it returns

    def __iter__(self):
        return self


class PrefetchDataLoader(DataLoader): ## undefined - sublcass of DataLoader
    """A fast data prefetch dataloader.

    Args:
        num_data_prefetch_queue (int): How many early data load queues.
        kwargs (dict): Other extended parameters.
    """

    def __init__(self, num_data_prefetch_queue: int, **kwargs) -> None: ## undefined - it takes num_data_prefetch_queue early data queues to use, and with kwargs it allows additional keyword arguments
        self.num_data_prefetch_queue = num_data_prefetch_queue ## undefined - stores the data queues 
        super(PrefetchDataLoader, self).__init__(**kwargs) ## undefined - ensures that the superclass initialization is performed before any additional initialization 

    def __iter__(self):
        return PrefetchGenerator(super().__iter__(), self.num_data_prefetch_queue)


class CPUPrefetcher: ## undefined 
    """Use the CPU side to accelerate data reading.

    Args:
        dataloader (DataLoader): Data loader. Combines a dataset and a sampler, and provides an iterable over the given dataset.
    """

    def __init__(self, dataloader) -> None: ## undefined - constructor of the class
        self.original_dataloader = dataloader ## undefined stores the reference to the original data loader
        self.data = iter(dataloader) ## undefined - an iterator created from the dataloader object

    def next(self): ## undefined - retrieves the next item from the iterator
        try:
            return next(self.data) ## undefined - returns the item
        except StopIteration: ## undefined - stops the iteration
            return None

    def reset(self): ## undefined - resets the iterator 
        self.data = iter(self.original_dataloader) ## undefined - allows iterating over the data loader again from the start

    def __len__(self) -> int: ## undefined - length function
        return len(self.original_dataloader) ## undefined - returns the nr of items in the dataset


class CUDAPrefetcher: ## undefined
    """Use the CUDA side to accelerate data reading.

    Args:
        dataloader (DataLoader): Data loader. Combines a dataset and a sampler, and provides an iterable over the given dataset.
        device (torch.device): Specify running device.
    """

    def __init__(self, dataloader, device: torch.device): ## undefined - constructor for the class 
        self.batch_data = None
        self.original_dataloader = dataloader ## undefined - stores the reference to the original data loader
        self.device = device ## undefined - stores the device object

        self.data = iter(dataloader) ## undefined - iterator from the dataLoader object
        self.stream = torch.cuda.Stream() ## undefined - CUDA stream used for asynchronous data transfer
        self.preload() ## undefined - calls the preload method 

    def preload(self): ## undefined - method that loads the next batch of data
        try:
            self.batch_data = next(self.data) ## undefined - retrieves the next batch
        except StopIteration: ## undefined - stops the iteration
            self.batch_data = None
            return None

        with torch.cuda.stream(self.stream):
            for k, v in self.batch_data.items():
                if torch.is_tensor(v):
                    self.batch_data[k] = self.batch_data[k].to(self.device, non_blocking=True)
  
    def next(self): ## undefined - iterator method
        torch.cuda.current_stream().wait_stream(self.stream) ## undefined - waits for the CUDA stream to finish 
        batch_data = self.batch_data ## undefined - assignes the current batcj of data 
        self.preload() ## undefined - loads the next batch of data
        return batch_data ## undefined - returns the current batch of data 

    def reset(self): ## undefined - reset method
        self.data = iter(self.original_dataloader) ## undefined - resets the iterator to the beginning
        self.preload() ## undefined - loads the first batch of data 

    def __len__(self) -> int: ## undefined - length method
        return len(self.original_dataloader) ## undefined - returns the number of batches in the data sheet 
