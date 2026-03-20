# -*- coding: utf-8 -*-
"""
工具函数模块
"""
import os
import yaml
import random
import numpy as np
import torch
from typing import Dict, List, Any, Optional


def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """
    加载配置文件
    """
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def set_seed(seed: int = 42) -> None:
    """
    设置随机种子
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """
    获取设备
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    epoch: int,
    metrics: Dict[str, float],
    path: str,
    is_best: bool = False
) -> None:
    """
    保存检查点
    """
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "metrics": metrics,
    }
    
    if is_best:
        best_path = os.path.join(os.path.dirname(path), "model_best.pth")
        torch.save(checkpoint, best_path)
    else:
        torch.save(checkpoint, path)


def load_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    device: Optional[torch.device] = None
) -> Dict[str, Any]:
    """
    加载检查点
    """
    if device is None:
        device = get_device()
    
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    
    if optimizer and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    if scheduler and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    
    return checkpoint


def create_dirs(config: Dict[str, Any]) -> None:
    """
    创建必要的目录
    """
    for dir_path in config["paths"].values():
        os.makedirs(dir_path, exist_ok=True)


def label_to_onehot(label: int, num_classes: int) -> torch.Tensor:
    """
    将标签转换为one-hot向量
    如果标签为-1，返回全零向量
    """
    if label == -1:
        return torch.zeros(num_classes, dtype=torch.float32)
    onehot = torch.zeros(num_classes, dtype=torch.float32)
    onehot[label] = 1.0
    return onehot


def parse_label(label_str: str) -> int:
    """
    解析标签字符串为数值
    """
    try:
        label = int(label_str)
        return label if label >= 0 else -1
    except (ValueError, TypeError):
        return -1


class AverageMeter:
    """
    计算并存储平均值和当前值
    """
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0


class MetricMeter:
    """
    多任务指标计算器
    """
    
    def __init__(self, num_tasks: int):
        self.num_tasks = num_tasks
        self.meters = [AverageMeter() for _ in range(num_tasks)]
    
    def reset(self):
        for meter in self.meters:
            meter.reset()
    
    def update(self, values: List[float], n: int = 1):
        for i, val in enumerate(values):
            self.meters[i].update(val, n)
    
    def get_averages(self) -> List[float]:
        return [meter.avg for meter in self.meters]
