import os
from os.path import isdir
import tarfile
import wget
import ssl
from pathlib import Path
from PIL import Image

from torch import tensor
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader

import random
from torch.utils.data import Subset

# Parameters
DATASETS_PATH = Path("./data")

DEFAULT_SIZE = 224

IMAGENET_MEAN = tensor([.485, .456, .406])  
IMAGENET_STD = tensor([.229, .224, .225]) 


class_links = {
    "coal_ad_test": "https://www.modelscope.cn/datasets/lyfjwp/CoalAD/resolve/master/coal_ad_test.tar",
    "coal_ad_train": "https://www.modelscope.cn/datasets/lyfjwp/CoalAD/resolve/master/coal_ad_train.tar",
}


def coal_classes():
    return [ 
        "coal_ad"
    ]


class CoalDataset:
    def __init__(
            self, 
            cls: str, 
            size: int = DEFAULT_SIZE, 
            train_data_ratio: float = 1.0
    ):
        assert cls in coal_classes()

        # Parameters
        self.cls = cls
        self.size = size  
        
        # Download data
        self.check_and_download_cls()

        self.train_ds = CoalTrainDataset(cls, size)
        self.test_ds = CoalTestDataset(cls, size)

        self.train_data_ratio = train_data_ratio
        if train_data_ratio < 1.0:
            self._sample_train_data()
        
    def _sample_train_data(self):
        """Randomly select specified proportion of training data"""
        total_size = len(self.train_ds)
        sample_size = int(total_size * self.train_data_ratio)
        
        # Randomly select indices
        indices = list(range(total_size))
        random.shuffle(indices)
        sampled_indices = indices[:sample_size]
        
        # Create subset
        self.train_ds = Subset(self.train_ds, sampled_indices)
        
        print(f"Sampled {sample_size} out of {total_size} training examples "
              f"({self.train_data_ratio * 100:.1f}%) for class '{self.cls}'")

    def check_and_download_cls(self):
        """
        If the expected dataset path is not found, 
        download the dataset inside /dataset.
        Assumes two tar files: {cls}_train.tar and {cls}_test.tar
        """

        if not isdir(DATASETS_PATH / self.cls):
            print(f"Class '{self.cls}' not found. Downloading...")
            
            ssl._create_default_https_context = ssl._create_unverified_context
            
            # 下载并解压train和test两个文件
            for suffix in ['_train', '_test']:
                tar_name = f"{self.cls}{suffix}.tar"
                link_key = f"{self.cls}{suffix}"
                
                if link_key in class_links:
                    print(f"Downloading {tar_name}...")
                    wget.download(class_links[link_key])
                    
                    print(f"\nExtracting {tar_name}...")
                    with tarfile.open(tar_name) as tar:
                        tar.extractall(DATASETS_PATH)
                    
                    os.remove(tar_name)
            
            print(f"\nDataset '{self.cls}' downloaded successfully")
        else:
            print(f"Dataset '{self.cls}' already exists")


    def get_datasets(self):
        """
        Returns as tuple:
        - train dataset (MVTecTrainDataset class)
        - test dataset (MVTecTestDataset class)
        """
        return self.train_ds, self.test_ds


    def get_dataloaders(self):
        """
        Returns as tuple:
        - train dataloader (torch.utils.data.DataLoader class)
        - test dataloader (torch.utils.data.DataLoader class)
        """
        return DataLoader(self.train_ds), DataLoader(self.test_ds)

    def _convert_image_to_rgb(image):
        return image.convert("RGB")


class CoalTrainDataset(ImageFolder):
    def __init__(
            self, 
            cls: str, 
            size: int, 
    ):
        transform = transforms.Compose([        
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD), 
        ])     

        # Parameters
        super().__init__(
                root = DATASETS_PATH / cls / "train",
                transform = transform )
        self.cls = cls
        self.size = size


class CoalTestDataset(ImageFolder):
    def __init__(
            self, 
            cls: str, 
            size: int, 
    ):

        transform = transforms.Compose([         # Image transform
            transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),  
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD), 
        ])
        target_transform = transforms.Compose([  # Mask transform
            transforms.Resize(size, interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor(),
        ])

        # Parameters
        super().__init__(
            root=DATASETS_PATH / cls / "test",
            transform = transform,
            target_transform = target_transform
        )

        self.cls = cls
        self.size = size


    def __getitem__(self, index):
        path, _ = self.samples[index]
        sample = self.loader(path)

        if "good" in path:                                      # Nominal image
            mask = Image.new('L', (self.size, self.size))       # L -> 8-bit pixels black and white
            sample_class = 0
        else:                                                   # Anomaly image
            mask_path = path.replace("test", "ground_truth")    # Change folder and goes into mask folder
            mask_path = mask_path.replace(".png", "_mask.png")  # Change extension required
            mask = self.loader(mask_path)                       # Load the mask
            sample_class = 1

        # Transformations 
        if self.transform is not None:
            sample = self.transform(sample)  

        if self.target_transform is not None:
            mask = self.target_transform(mask)

        return sample, mask[:1], sample_class