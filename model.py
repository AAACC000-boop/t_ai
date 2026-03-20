import torch
import torch.nn as nn
import timm


class MultiTaskTongueModel(nn.Module):
    """多任务舌象分类模型"""
    
    def __init__(self, task_configs, pretrained=True, dropout=0.3):
        """
        Args:
            task_configs: 任务配置列表，每个元素是num_classes
            pretrained: 是否使用预训练权重
            dropout: dropout比率
        """
        super(MultiTaskTongueModel, self).__init__()
        
        self.task_configs = task_configs
        self.num_tasks = len(task_configs)
        
        # 使用ConvNeXtV2 Base作为backbone
        self.backbone = timm.create_model(
            'convnextv2_base',
            pretrained=pretrained,
            num_classes=0,  # 移除原始分类头
            global_pool='avg'
        )
        
        # 获取特征维度
        self.feature_dim = self.backbone.num_features
        
        # 为每个任务创建独立的分类头
        self.heads = nn.ModuleList()
        for num_classes in task_configs:
            head = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(self.feature_dim, 512),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(512, num_classes)
            )
            self.heads.append(head)
    
    def forward(self, x):
        """
        Args:
            x: 输入图像 [batch_size, 3, H, W]
        Returns:
            logits_list: 每个任务的logits列表，长度为num_tasks
                        每个元素形状为 [batch_size, num_classes_i]
        """
        # 提取特征
        features = self.backbone(x)  # [batch_size, feature_dim]
        
        # 每个任务独立预测
        logits_list = []
        for head in self.heads:
            logits = head(features)  # [batch_size, num_classes_i]
            logits_list.append(logits)
        
        return logits_list
    
    def get_features(self, x):
        """获取特征向量（用于可视化分析）"""
        return self.backbone(x)


def create_model(task_configs, pretrained=True, dropout=0.3):
    """创建模型"""
    model = MultiTaskTongueModel(
        task_configs=task_configs,
        pretrained=pretrained,
        dropout=dropout
    )
    return model
