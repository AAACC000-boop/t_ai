import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


def read_csv_with_encoding(csv_path):
    """尝试多种编码读取CSV文件"""
    encodings = ['utf-8', 'gbk', 'gb2312', 'gb18030', 'latin1']
    for encoding in encodings:
        try:
            df = pd.read_csv(csv_path, encoding=encoding)
            print(f"Successfully read CSV with encoding: {encoding}")
            return df
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Could not read CSV file with any encoding: {csv_path}")


TASK_CONFIG = {
    '舌体颜色': {'num_classes': 5, 'classes': ['淡红', '白', '红', '绛', '紫']},
    '舌体特征': {'num_classes': 2, 'classes': ['老', '嫩']},
    '舌体形状': {'num_classes': 3, 'classes': ['胖大', '瘦小', '正常']},
    '齿痕': {'num_classes': 2, 'classes': ['无齿痕', '有齿痕']},
    '点刺': {'num_classes': 2, 'classes': ['无点刺', '有点刺']},
    '裂纹': {'num_classes': 2, 'classes': ['无裂纹', '有裂纹']},
    '舌苔特征': {'num_classes': 3, 'classes': ['舌苔厚', '舌苔薄', '舌苔正常']},
    '滑苔': {'num_classes': 2, 'classes': ['无滑苔', '有滑苔']},
    '糙苔': {'num_classes': 2, 'classes': ['无糙苔', '有糙苔']},
    '剥落苔': {'num_classes': 2, 'classes': ['无剥落苔', '有剥落苔']},
    '腐苔': {'num_classes': 2, 'classes': ['无腐苔', '有腐苔']},
    '腻苔': {'num_classes': 2, 'classes': ['无腻苔', '有腻苔']},
    '舌苔颜色': {'num_classes': 4, 'classes': ['白', '黄', '灰', '黑']},
    '舌尖红': {'num_classes': 2, 'classes': ['无舌尖红', '有舌尖红']},
    '舌尖凹陷': {'num_classes': 2, 'classes': ['无舌尖凹陷', '有舌尖凹陷']},
}

TASK_NAMES = list(TASK_CONFIG.keys())


def get_transforms(train=True, image_size=224):
    """获取数据增强变换"""
    if train:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


class TongueDataset(Dataset):
    """舌象数据集"""
    
    def __init__(self, csv_path, image_dir, transform=None, task_names=None):
        """
        Args:
            csv_path: CSV文件路径
            image_dir: 图像目录路径
            transform: 图像变换
            task_names: 任务名称列表，如果为None则使用默认顺序
        """
        self.df = read_csv_with_encoding(csv_path)
        self.image_dir = image_dir
        self.transform = transform
        
        # 使用默认任务顺序
        if task_names is None:
            self.task_names = TASK_NAMES
        else:
            self.task_names = task_names
        
        # 获取CSV列名（第一列是图片名，后面是15个任务标签）
        self.columns = self.df.columns.tolist()
        self.image_col = self.columns[0]
        self.label_cols = self.columns[1:16]  # 15个标签列
        
        # 验证任务数量
        assert len(self.label_cols) == 15, f"Expected 15 label columns, got {len(self.label_cols)}"
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        """
        Returns:
            image: 变换后的图像张量
            labels: 标签列表，每个元素是one-hot向量或全零向量（缺失时）
            mask: 掩码列表，1表示标签有效，0表示缺失
        """
        row = self.df.iloc[idx]
        
        # 加载图像
        image_name = row[self.image_col]
        image_path = os.path.join(self.image_dir, image_name)
        
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # 返回一个空白图像
            image = Image.new('RGB', (224, 224), (0, 0, 0))
        
        if self.transform:
            image = self.transform(image)
        
        # 处理标签
        labels = []
        masks = []
        
        for i, col in enumerate(self.label_cols):
            task_name = self.task_names[i]
            num_classes = TASK_CONFIG[task_name]['num_classes']
            
            label_value = row[col]
            
            # 处理缺失值
            if pd.isna(label_value) or label_value == -1:
                # 缺失标签：one-hot全零，mask为0
                one_hot = torch.zeros(num_classes, dtype=torch.float32)
                mask = 0
            else:
                # 有效标签：转换为one-hot
                label_idx = int(label_value)
                one_hot = torch.zeros(num_classes, dtype=torch.float32)
                if 0 <= label_idx < num_classes:
                    one_hot[label_idx] = 1.0
                mask = 1
            
            labels.append(one_hot)
            masks.append(mask)
        
        return image, labels, torch.tensor(masks, dtype=torch.float32), image_name


def collate_fn(batch):
    """自定义collate函数处理多任务标签"""
    images = torch.stack([item[0] for item in batch])
    
    # 收集每个任务的标签
    num_tasks = len(batch[0][1])
    labels_list = []
    for task_idx in range(num_tasks):
        task_labels = torch.stack([item[1][task_idx] for item in batch])
        labels_list.append(task_labels)
    
    masks = torch.stack([item[2] for item in batch])
    image_names = [item[3] for item in batch]
    
    return images, labels_list, masks, image_names
