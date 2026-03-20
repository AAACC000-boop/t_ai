import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from typing import List, Tuple, Optional, Dict
import numpy as np

import config


class TongueDataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        image_dir: str,
        transform: Optional[transforms.Compose] = None,
        is_train: bool = True
    ):
        self.df = pd.read_csv(csv_path, encoding='gbk')
        self.image_dir = image_dir
        self.transform = transform
        self.is_train = is_train
        
        self.image_col = self.df.columns[0]
        self.label_cols = self.df.columns[1:1 + config.NUM_TASKS]
        
        self._validate_data()
    
    def _validate_data(self):
        valid_indices = []
        for idx in range(len(self.df)):
            img_name = self.df.iloc[idx, 0]
            img_path = os.path.join(self.image_dir, img_name)
            if os.path.exists(img_path):
                valid_indices.append(idx)
        
        if len(valid_indices) < len(self.df):
            print(f"Warning: {len(self.df) - len(valid_indices)} images not found")
            self.df = self.df.iloc[valid_indices].reset_index(drop=True)
    
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, List[torch.Tensor], str]:
        row = self.df.iloc[idx]
        img_name = row[self.image_col]
        img_path = os.path.join(self.image_dir, img_name)
        
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        labels = []
        for i, col in enumerate(self.label_cols):
            label_val = row[col]
            num_classes = config.NUM_CLASSES_PER_TASK[i]
            
            if pd.isna(label_val) or label_val == config.MISSING_LABEL:
                one_hot = torch.zeros(num_classes, dtype=torch.float32)
            else:
                label_int = int(label_val)
                if 0 <= label_int < num_classes:
                    one_hot = torch.zeros(num_classes, dtype=torch.float32)
                    one_hot[label_int] = 1.0
                else:
                    one_hot = torch.zeros(num_classes, dtype=torch.float32)
            
            labels.append(one_hot)
        
        return image, labels, img_name


def get_transforms(is_train: bool = True) -> transforms.Compose:
    if is_train:
        return transforms.Compose([
            transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD)
        ])
    else:
        return transforms.Compose([
            transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD)
        ])


def get_dataloaders(
    csv_path: str = str(config.CSV_PATH),
    image_dir: str = str(config.IMAGE_DIR),
    batch_size: int = config.BATCH_SIZE,
    val_split: float = config.VAL_SPLIT,
    num_workers: int = config.NUM_WORKERS,
    seed: int = config.SEED
) -> Tuple[DataLoader, DataLoader]:
    np.random.seed(seed)
    
    full_dataset = TongueDataset(
        csv_path=csv_path,
        image_dir=image_dir,
        transform=None,
        is_train=True
    )
    
    total_size = len(full_dataset)
    indices = list(range(total_size))
    np.random.shuffle(indices)
    
    val_size = int(total_size * val_split)
    train_indices = indices[val_size:]
    val_indices = indices[:val_size]
    
    train_df = full_dataset.df.iloc[train_indices].reset_index(drop=True)
    val_df = full_dataset.df.iloc[val_indices].reset_index(drop=True)
    
    train_dataset = TongueDatasetFromDF(
        train_df, image_dir, get_transforms(is_train=True), is_train=True
    )
    val_dataset = TongueDatasetFromDF(
        val_df, image_dir, get_transforms(is_train=False), is_train=False
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    return train_loader, val_loader


class TongueDatasetFromDF(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        image_dir: str,
        transform: Optional[transforms.Compose] = None,
        is_train: bool = True
    ):
        self.df = df
        self.image_dir = image_dir
        self.transform = transform
        self.is_train = is_train
        
        self.image_col = self.df.columns[0]
        self.label_cols = self.df.columns[1:1 + config.NUM_TASKS]
    
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, List[torch.Tensor], str]:
        row = self.df.iloc[idx]
        img_name = row[self.image_col]
        img_path = os.path.join(self.image_dir, img_name)
        
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        labels = []
        for i, col in enumerate(self.label_cols):
            label_val = row[col]
            num_classes = config.NUM_CLASSES_PER_TASK[i]
            
            if pd.isna(label_val) or label_val == config.MISSING_LABEL:
                one_hot = torch.zeros(num_classes, dtype=torch.float32)
            else:
                label_int = int(label_val)
                if 0 <= label_int < num_classes:
                    one_hot = torch.zeros(num_classes, dtype=torch.float32)
                    one_hot[label_int] = 1.0
                else:
                    one_hot = torch.zeros(num_classes, dtype=torch.float32)
            
            labels.append(one_hot)
        
        return image, labels, img_name


def get_full_dataset_loader(
    csv_path: str = str(config.CSV_PATH),
    image_dir: str = str(config.IMAGE_DIR),
    batch_size: int = config.BATCH_SIZE,
    num_workers: int = config.NUM_WORKERS
) -> DataLoader:
    dataset = TongueDataset(
        csv_path=csv_path,
        image_dir=image_dir,
        transform=get_transforms(is_train=False),
        is_train=False
    )
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return loader
