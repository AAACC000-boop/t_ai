# -*- coding: utf-8 -*-
"""
数据集加载和预处理模块
"""
import os
import pandas as pd
import numpy as np
from PIL import Image
from typing import Dict, List, Tuple, Any, Optional

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms

from src.utils import label_to_onehot, parse_label


class TongueDataset(Dataset):
    """
    舌象数据集
    """
    
    def __init__(
        self,
        image_dir: str,
        csv_file: str,
        transform: Optional[transforms.Compose] = None,
        num_classes_list: Optional[List[int]] = None,
        mode: str = "train"
    ):
        self.image_dir = image_dir
        self.transform = transform
        self.mode = mode
        
        # 默认类别数
        if num_classes_list is None:
            self.num_classes_list = [5, 2, 3, 2, 2, 2, 3, 2, 2, 2, 2, 2, 4, 2, 2]
        else:
            self.num_classes_list = num_classes_list
        
        self.num_tasks = len(self.num_classes_list)
        
        # 读取CSV文件（尝试不同编码）
        try:
            self.data = pd.read_csv(csv_file, encoding="utf-8")
        except:
            try:
                self.data = pd.read_csv(csv_file, encoding="gbk")
            except:
                self.data = pd.read_csv(csv_file, encoding="gb2312")
        
        # 获取图片文件名列表
        self.image_files = self.data.iloc[:, 0].tolist()
        
        # 解析标签
        self.labels = self._parse_labels()
    
    def _parse_labels(self) -> List[List[int]]:
        """
        解析CSV中的标签
        """
        labels = []
        
        for _, row in self.data.iterrows():
            # 获取15个任务的标签
            task_labels = []
            for i in range(1, 16):  # 第1列是文件名，2-16列是标签
                label_str = str(row[i]) if pd.notna(row[i]) else "-1"
                label = parse_label(label_str)
                task_labels.append(label)
            labels.append(task_labels)
        
        return labels
    
    def __len__(self) -> int:
        return len(self.image_files)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, List[torch.Tensor], str]:
        """
        获取样本
        返回: (图像张量, 15个任务的one-hot标签列表, 文件名)
        """
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)
        
        # 读取图像
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"无法读取图像 {img_path}: {e}")
            # 返回一个占位符
            image = Image.new("RGB", (224, 224))
        
        # 数据增强
        if self.transform:
            image = self.transform(image)
        
        # 获取标签并转换为one-hot
        raw_labels = self.labels[idx]
        onehot_labels = [
            label_to_onehot(label, num_classes)
            for label, num_classes in zip(raw_labels, self.num_classes_list)
        ]
        
        return image, onehot_labels, img_name


def get_transform(aug_config: Dict[str, Any]) -> transforms.Compose:
    """
    根据配置创建数据增强变换
    """
    transform_list = []
    
    for name, params in aug_config.items():
        if params is None:
            transform_list.append(getattr(transforms, name)())
        else:
            transform_list.append(getattr(transforms, name)(**params))
    
    return transforms.Compose(transform_list)


def create_dataloaders(config: Dict[str, Any]) -> Tuple[DataLoader, DataLoader]:
    """
    创建训练和验证数据加载器
    """
    data_config = config["data"]
    aug_config = config["augmentation"]
    task_config = config["task"]
    
    # 创建数据增强
    train_transform = get_transform(aug_config["train"])
    val_transform = get_transform(aug_config["val"])
    
    # 创建完整数据集
    full_dataset = TongueDataset(
        image_dir=data_config["image_dir"],
        csv_file=data_config["csv_file"],
        num_classes_list=task_config["num_classes"],
        mode="train"
    )
    
    # 划分训练集和验证集
    train_size = int(data_config["train_ratio"] * len(full_dataset))
    val_size = len(full_dataset) - train_size
    
    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(data_config["random_seed"])
    )
    
    # 分别设置transform
    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = val_transform
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=data_config["batch_size"],
        shuffle=True,
        num_workers=data_config["num_workers"],
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=data_config["batch_size"],
        shuffle=False,
        num_workers=data_config["num_workers"],
        pin_memory=True
    )
    
    return train_loader, val_loader


def create_test_dataloader(config: Dict[str, Any], csv_file: Optional[str] = None) -> DataLoader:
    """
    创建测试数据加载器
    """
    data_config = config["data"]
    aug_config = config["augmentation"]
    task_config = config["task"]
    
    if csv_file is None:
        csv_file = data_config["csv_file"]
    
    test_transform = get_transform(aug_config["val"])
    
    test_dataset = TongueDataset(
        image_dir=data_config["image_dir"],
        csv_file=csv_file,
        transform=test_transform,
        num_classes_list=task_config["num_classes"],
        mode="test"
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=data_config["batch_size"],
        shuffle=False,
        num_workers=data_config["num_workers"],
        pin_memory=True
    )
    
    return test_loader


class InferenceDataset(Dataset):
    """
    推理用数据集（无标签）
    """
    
    def __init__(
        self,
        image_dir: str,
        image_files: Optional[List[str]] = None,
        transform: Optional[transforms.Compose] = None
    ):
        self.image_dir = image_dir
        self.transform = transform
        
        if image_files is None:
            # 获取目录下所有图片文件
            self.image_files = [
                f for f in os.listdir(image_dir)
                if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
            ]
        else:
            self.image_files = image_files
    
    def __len__(self) -> int:
        return len(self.image_files)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)
        
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"无法读取图像 {img_path}: {e}")
            image = Image.new("RGB", (224, 224))
        
        if self.transform:
            image = self.transform(image)
        
        return image, img_name


def create_inference_dataloader(
    config: Dict[str, Any],
    image_dir: Optional[str] = None,
    image_files: Optional[List[str]] = None
) -> DataLoader:
    """
    创建推理用数据加载器
    """
    data_config = config["data"]
    aug_config = config["augmentation"]
    
    if image_dir is None:
        image_dir = data_config["image_dir"]
    
    test_transform = get_transform(aug_config["val"])
    
    inference_dataset = InferenceDataset(
        image_dir=image_dir,
        image_files=image_files,
        transform=test_transform
    )
    
    inference_loader = DataLoader(
        inference_dataset,
        batch_size=data_config["batch_size"],
        shuffle=False,
        num_workers=data_config["num_workers"],
        pin_memory=True
    )
    
    return inference_loader
