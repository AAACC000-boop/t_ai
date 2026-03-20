import torch
import torch.nn as nn
from typing import List
import timm

import config


class MultiHeadTongueModel(nn.Module):
    def __init__(
        self,
        model_name: str = "convnextv2_base",
        pretrained: bool = True,
        num_classes_per_task: List[int] = None
    ):
        super().__init__()
        
        if num_classes_per_task is None:
            num_classes_per_task = config.NUM_CLASSES_PER_TASK
        
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0
        )
        
        in_features = self.backbone.num_features
        
        self.heads = nn.ModuleList([
            nn.Linear(in_features, num_classes)
            for num_classes in num_classes_per_task
        ])
        
        self.num_tasks = len(num_classes_per_task)
    
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        features = self.backbone(x)
        
        outputs = [head(features) for head in self.heads]
        
        return outputs
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


class MultiTaskLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion = nn.BCEWithLogitsLoss(reduction='none')
    
    def forward(
        self,
        predictions: List[torch.Tensor],
        targets: List[torch.Tensor]
    ) -> torch.Tensor:
        total_loss = 0.0
        num_valid_tasks = 0
        
        for pred, target in zip(predictions, targets):
            valid_mask = target.sum(dim=1) > 0
            
            if valid_mask.sum() > 0:
                loss = self.criterion(pred[valid_mask], target[valid_mask])
                loss = loss.mean()
                total_loss += loss
                num_valid_tasks += 1
        
        if num_valid_tasks > 0:
            return total_loss / num_valid_tasks
        else:
            return torch.tensor(0.0, device=predictions[0].device)


def load_model(checkpoint_path: str = None, device: str = 'cuda') -> MultiHeadTongueModel:
    model = MultiHeadTongueModel()
    
    if checkpoint_path and torch.cuda.is_available():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print(f"Loaded checkpoint from {checkpoint_path}")
    
    model = model.to(device)
    return model


def save_model(
    model: MultiHeadTongueModel,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_f1: float,
    checkpoint_path: str
):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_f1': best_f1
    }, checkpoint_path)
    print(f"Saved checkpoint to {checkpoint_path}")
