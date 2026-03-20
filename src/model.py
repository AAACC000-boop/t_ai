# -*- coding: utf-8 -*-
"""
多任务舌象识别模型
"""
import torch
import torch.nn as nn
import timm
from typing import List, Dict, Any


class MultiTaskConvNeXtV2(nn.Module):
    """
    基于ConvNeXtV2的多任务分类模型
    """
    
    def __init__(
        self,
        model_name: str = "convnextv2_base.fcmae_ft_in22k_in1k",
        pretrained: bool = True,
        num_classes_list: List[int] = None,
        dropout: float = 0.5
    ):
        super().__init__()
        
        if num_classes_list is None:
            num_classes_list = [5, 2, 3, 2, 2, 2, 3, 2, 2, 2, 2, 2, 4, 2, 2]
        
        self.num_classes_list = num_classes_list
        self.num_tasks = len(num_classes_list)
        
        # 加载ConvNeXtV2骨干网络
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,  # 移除原始分类头
            drop_rate=dropout,
        )
        
        # 获取特征维度
        self.feature_dim = self.backbone.num_features
        
        # 创建15个独立的分类头
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(self.feature_dim),
                nn.Linear(self.feature_dim, num_classes)
            ) for num_classes in num_classes_list
        ])
    
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        前向传播
        返回: 15个任务的logits列表，每个形状为[batch_size, num_classes_i]
        """
        # 提取特征
        features = self.backbone(x)
        
        # 每个任务独立分类
        logits = [head(features) for head in self.heads]
        
        return logits
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        提取特征向量（用于可视化）
        """
        return self.backbone(x)


class MultiTaskLoss(nn.Module):
    """
    多任务损失函数
    对每个任务使用交叉熵损失，忽略缺失标签（-1）
    """
    
    def __init__(self, num_classes_list: List[int] = None, weight: List[float] = None):
        super().__init__()
        
        if num_classes_list is None:
            num_classes_list = [5, 2, 3, 2, 2, 2, 3, 2, 2, 2, 2, 2, 4, 2, 2]
        
        self.num_classes_list = num_classes_list
        self.num_tasks = len(num_classes_list)
        self.weight = weight if weight is not None else [1.0] * self.num_tasks
        
        # 每个任务使用交叉熵损失
        self.loss_fns = nn.ModuleList([
            nn.CrossEntropyLoss(reduction="none") for _ in range(self.num_tasks)
        ])
    
    def forward(
        self,
        logits_list: List[torch.Tensor],
        targets_list: List[torch.Tensor]
    ) -> Dict[str, Any]:
        """
        计算多任务损失
        Args:
            logits_list: 15个任务的logits列表
            targets_list: 15个任务的目标值列表 (one-hot向量或标签索引)
        Returns:
            total_loss: 总损失
            losses: 各任务损失字典
        """
        losses = {}
        total_loss = 0.0
        valid_task_count = 0
        
        for task_idx, (logits, targets) in enumerate(zip(logits_list, targets_list)):
            # 获取有效样本掩码（标签不为-1）
            if targets.dim() > 1:
                # one-hot格式，全零向量表示缺失
                valid_mask = targets.sum(dim=1) > 0
                # 转换为类别索引
                targets = targets.argmax(dim=1)
            else:
                # 标签索引格式
                valid_mask = targets != -1
            
            num_valid = valid_mask.sum().item()
            if num_valid == 0:
                losses[f"task_{task_idx}"] = torch.tensor(0.0, device=logits.device)
                continue
            
            # 只计算有效样本的损失
            valid_logits = logits[valid_mask]
            valid_targets = targets[valid_mask]
            
            task_loss = self.loss_fns[task_idx](valid_logits, valid_targets)
            task_loss = task_loss.mean() * self.weight[task_idx]
            
            losses[f"task_{task_idx}"] = task_loss
            total_loss += task_loss
            valid_task_count += 1
        
        # 平均损失
        if valid_task_count > 0:
            total_loss = total_loss / valid_task_count
        
        losses["total"] = total_loss
        
        return losses


def create_model(config: Dict[str, Any]) -> nn.Module:
    """
    根据配置创建模型
    """
    model_config = config["model"]
    task_config = config["task"]
    
    model = MultiTaskConvNeXtV2(
        model_name=model_config["name"],
        pretrained=model_config["pretrained"],
        num_classes_list=task_config["num_classes"],
        dropout=model_config["dropout"]
    )
    
    return model


def create_loss_fn(config: Dict[str, Any]) -> nn.Module:
    """
    创建损失函数
    """
    task_config = config["task"]
    
    loss_fn = MultiTaskLoss(
        num_classes_list=task_config["num_classes"]
    )
    
    return loss_fn
