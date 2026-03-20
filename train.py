import os
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import json

from dataset import TongueDataset, get_transforms, collate_fn, TASK_CONFIG, TASK_NAMES
from model import create_model


def set_seed(seed=42):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def masked_cross_entropy_loss(logits, targets, masks):
    """
    带掩码的交叉熵损失，忽略缺失标签
    Args:
        logits: [batch_size, num_classes]
        targets: [batch_size, num_classes] one-hot编码
        masks: [batch_size] 1表示有效，0表示缺失
    """
    # 将one-hot转换为类别索引
    target_indices = torch.argmax(targets, dim=1)
    
    # 计算交叉熵损失（不减均值）
    loss = nn.functional.cross_entropy(logits, target_indices, reduction='none')
    
    # 应用掩码
    masked_loss = loss * masks
    
    # 只对有效标签求平均
    if masks.sum() > 0:
        return masked_loss.sum() / masks.sum()
    else:
        return torch.tensor(0.0, device=logits.device)


def train_epoch(model, dataloader, optimizer, device, task_names):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    task_losses = {name: 0.0 for name in task_names}
    task_counts = {name: 0 for name in task_names}
    
    pbar = tqdm(dataloader, desc='Training')
    for images, labels_list, masks, _ in pbar:
        images = images.to(device)
        labels_list = [labels.to(device) for labels in labels_list]
        masks = masks.to(device)
        
        optimizer.zero_grad()
        
        # 前向传播
        logits_list = model(images)
        
        # 计算每个任务的损失
        batch_loss = 0
        for i, (logits, labels, task_name) in enumerate(zip(logits_list, labels_list, task_names)):
            task_mask = masks[:, i]
            loss = masked_cross_entropy_loss(logits, labels, task_mask)
            batch_loss += loss
            
            if task_mask.sum() > 0:
                task_losses[task_name] += loss.item()
                task_counts[task_name] += 1
        
        # 反向传播
        batch_loss.backward()
        optimizer.step()
        
        total_loss += batch_loss.item()
        pbar.set_postfix({'loss': batch_loss.item()})
    
    # 计算平均损失
    avg_loss = total_loss / len(dataloader)
    for task_name in task_names:
        if task_counts[task_name] > 0:
            task_losses[task_name] /= task_counts[task_name]
    
    return avg_loss, task_losses


def validate(model, dataloader, device, task_names):
    """验证"""
    model.eval()
    total_loss = 0
    task_losses = {name: 0.0 for name in task_names}
    task_counts = {name: 0 for name in task_names}
    
    # 统计每个任务的准确率
    task_correct = {name: 0 for name in task_names}
    task_total = {name: 0 for name in task_names}
    
    with torch.no_grad():
        for images, labels_list, masks, _ in tqdm(dataloader, desc='Validating'):
            images = images.to(device)
            labels_list = [labels.to(device) for labels in labels_list]
            masks = masks.to(device)
            
            # 前向传播
            logits_list = model(images)
            
            # 计算损失和准确率
            batch_loss = 0
            for i, (logits, labels, task_name) in enumerate(zip(logits_list, labels_list, task_names)):
                task_mask = masks[:, i]
                loss = masked_cross_entropy_loss(logits, labels, task_mask)
                batch_loss += loss
                
                if task_mask.sum() > 0:
                    task_losses[task_name] += loss.item()
                    task_counts[task_name] += 1
                
                # 计算准确率
                preds = torch.argmax(logits, dim=1)
                targets = torch.argmax(labels, dim=1)
                
                for j in range(len(task_mask)):
                    if task_mask[j] > 0:
                        task_total[task_name] += 1
                        if preds[j] == targets[j]:
                            task_correct[task_name] += 1
            
            total_loss += batch_loss.item()
    
    # 计算平均损失和准确率
    avg_loss = total_loss / len(dataloader)
    
    for task_name in task_names:
        if task_counts[task_name] > 0:
            task_losses[task_name] /= task_counts[task_name]
    
    task_accs = {}
    for task_name in task_names:
        if task_total[task_name] > 0:
            task_accs[task_name] = task_correct[task_name] / task_total[task_name]
        else:
            task_accs[task_name] = 0.0
    
    overall_acc = sum(task_accs.values()) / len(task_accs)
    
    return avg_loss, task_losses, task_accs, overall_acc


def main(args):
    # 设置随机种子
    set_seed(args.seed)
    
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 任务配置
    task_configs = [TASK_CONFIG[name]['num_classes'] for name in TASK_NAMES]
    print(f"Tasks: {TASK_NAMES}")
    print(f"Task configs: {task_configs}")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 数据变换
    train_transform = get_transforms(train=True, image_size=args.image_size)
    val_transform = get_transforms(train=False, image_size=args.image_size)
    
    # 加载数据集
    full_dataset = TongueDataset(
        csv_path=args.csv_path,
        image_dir=args.image_dir,
        transform=train_transform
    )
    
    # 划分训练集和验证集
    dataset_size = len(full_dataset)
    val_size = int(dataset_size * args.val_split)
    train_size = dataset_size - val_size
    
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed)
    )
    
    # 验证集使用不同的变换
    val_dataset.dataset.transform = val_transform
    
    print(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}")
    
    # 数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    # 创建模型
    model = create_model(
        task_configs=task_configs,
        pretrained=True,
        dropout=args.dropout
    )
    model = model.to(device)
    
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # 优化器和学习率调度器
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # 训练循环
    best_val_acc = 0.0
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_acc': [],
        'task_accs': {name: [] for name in TASK_NAMES}
    }
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        print("-" * 50)
        
        # 训练
        train_loss, train_task_losses = train_epoch(
            model, train_loader, optimizer, device, TASK_NAMES
        )
        
        # 验证
        val_loss, val_task_losses, val_task_accs, val_overall_acc = validate(
            model, val_loader, device, TASK_NAMES
        )
        
        # 更新学习率
        scheduler.step()
        
        # 记录历史
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_overall_acc)
        for task_name in TASK_NAMES:
            history['task_accs'][task_name].append(val_task_accs[task_name])
        
        # 打印结果
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_overall_acc:.4f}")
        print("Task Accuracies:")
        for task_name in TASK_NAMES:
            print(f"  {task_name}: {val_task_accs[task_name]:.4f}")
        
        # 保存最佳模型
        if val_overall_acc > best_val_acc:
            best_val_acc = val_overall_acc
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_overall_acc,
                'task_accs': val_task_accs,
                'args': vars(args)
            }
            torch.save(checkpoint, os.path.join(args.output_dir, 'best_model.pth'))
            print(f"Saved best model with acc: {val_overall_acc:.4f}")
        
        # 保存最新模型
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_acc': val_overall_acc,
            'task_accs': val_task_accs,
            'args': vars(args)
        }
        torch.save(checkpoint, os.path.join(args.output_dir, 'latest_model.pth'))
    
    # 保存训练历史
    with open(os.path.join(args.output_dir, 'history.json'), 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=2, ensure_ascii=False)
    
    print(f"\nTraining completed! Best val acc: {best_val_acc:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Tongue Classification Model')
    
    # 数据参数
    parser.add_argument('--csv_path', type=str, default='result.csv', help='CSV文件路径')
    parser.add_argument('--image_dir', type=str, default='images', help='图像目录')
    parser.add_argument('--output_dir', type=str, default='outputs', help='输出目录')
    
    # 模型参数
    parser.add_argument('--image_size', type=int, default=224, help='图像大小')
    parser.add_argument('--dropout', type=float, default=0.3, help='dropout比率')
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=16, help='批次大小')
    parser.add_argument('--lr', type=float, default=1e-4, help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='权重衰减')
    parser.add_argument('--val_split', type=float, default=0.2, help='验证集比例')
    parser.add_argument('--num_workers', type=int, default=4, help='数据加载线程数')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    
    args = parser.parse_args()
    main(args)
