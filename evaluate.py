import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image, ImageDraw, ImageFont

from dataset import TongueDataset, get_transforms, collate_fn, TASK_CONFIG, TASK_NAMES
from model import create_model


def load_model(checkpoint_path, task_configs, device):
    """加载训练好的模型"""
    model = create_model(task_configs=task_configs, pretrained=False)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    return model


def evaluate_model(model, dataloader, device, task_names):
    """评估模型"""
    model.eval()
    
    all_predictions = {name: [] for name in task_names}
    all_targets = {name: [] for name in task_names}
    all_image_names = []
    
    with torch.no_grad():
        for images, labels_list, masks, image_names in tqdm(dataloader, desc='Evaluating'):
            images = images.to(device)
            labels_list = [labels.to(device) for labels in labels_list]
            masks = masks.to(device)
            
            # 前向传播
            logits_list = model(images)
            
            # 收集预测和真实标签
            for i, (logits, labels, task_name) in enumerate(zip(logits_list, labels_list, task_names)):
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                targets = torch.argmax(labels, dim=1).cpu().numpy()
                task_mask = masks[:, i].cpu().numpy()
                
                # 只收集有效标签
                for j in range(len(task_mask)):
                    if task_mask[j] > 0:
                        all_predictions[task_name].append(preds[j])
                        all_targets[task_name].append(targets[j])
            
            all_image_names.extend(image_names)
    
    return all_predictions, all_targets, all_image_names


def compute_metrics(predictions, targets, task_names):
    """计算每个任务的评估指标"""
    metrics = {}
    
    for task_name in task_names:
        preds = np.array(predictions[task_name])
        targs = np.array(targets[task_name])
        
        if len(preds) == 0:
            metrics[task_name] = {
                'accuracy': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0,
                'support': 0
            }
            continue
        
        # 计算指标
        acc = accuracy_score(targs, preds)
        
        # 对于多分类，使用macro平均
        num_classes = TASK_CONFIG[task_name]['num_classes']
        if num_classes > 2:
            prec = precision_score(targs, preds, average='macro', zero_division=0)
            rec = recall_score(targs, preds, average='macro', zero_division=0)
            f1 = f1_score(targs, preds, average='macro', zero_division=0)
        else:
            prec = precision_score(targs, preds, average='binary', zero_division=0)
            rec = recall_score(targs, preds, average='binary', zero_division=0)
            f1 = f1_score(targs, preds, average='binary', zero_division=0)
        
        metrics[task_name] = {
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1': f1,
            'support': len(preds)
        }
    
    # 计算总体平均指标
    overall_acc = np.mean([metrics[name]['accuracy'] for name in task_names])
    overall_f1 = np.mean([metrics[name]['f1'] for name in task_names])
    
    metrics['overall'] = {
        'accuracy': overall_acc,
        'f1': overall_f1
    }
    
    return metrics


def save_confusion_matrices(predictions, targets, task_names, output_dir):
    """保存每个任务的混淆矩阵"""
    cm_dir = os.path.join(output_dir, 'confusion_matrices')
    os.makedirs(cm_dir, exist_ok=True)
    
    for task_name in task_names:
        preds = np.array(predictions[task_name])
        targs = np.array(targets[task_name])
        
        if len(preds) == 0:
            continue
        
        num_classes = TASK_CONFIG[task_name]['num_classes']
        class_names = TASK_CONFIG[task_name]['classes']
        
        # 计算混淆矩阵
        cm = confusion_matrix(targs, preds, labels=list(range(num_classes)))
        
        # 保存为CSV
        cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
        cm_df.to_csv(os.path.join(cm_dir, f'{task_name}_confusion_matrix.csv'), encoding='utf-8-sig')
        
        # 绘制热力图
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names)
        plt.title(f'{task_name} - Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(os.path.join(cm_dir, f'{task_name}_confusion_matrix.png'), dpi=150)
        plt.close()


def save_predictions_csv(all_predictions, all_targets, image_names, task_names, output_dir):
    """保存预测结果CSV"""
    # 创建DataFrame
    data = {'image_name': image_names}
    
    for task_name in task_names:
        # 将预测扩展到所有样本（缺失的用-1填充）
        task_preds = [-1] * len(image_names)
        task_targets = [-1] * len(image_names)
        
        for i, (pred, target) in enumerate(zip(all_predictions[task_name], all_targets[task_name])):
            if i < len(image_names):
                task_preds[i] = pred
                task_targets[i] = target
        
        data[f'{task_name}_pred'] = task_preds
        data[f'{task_name}_true'] = task_targets
    
    df = pd.DataFrame(data)
    df.to_csv(os.path.join(output_dir, 'predictions.csv'), index=False, encoding='utf-8-sig')


def visualize_predictions(model, dataloader, device, task_names, output_dir, num_samples=20):
    """可视化预测结果"""
    vis_dir = os.path.join(output_dir, 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)
    
    model.eval()
    
    samples_collected = 0
    results = []
    
    with torch.no_grad():
        for images, labels_list, masks, image_names in dataloader:
            if samples_collected >= num_samples:
                break
            
            images = images.to(device)
            logits_list = model(images)
            
            # 获取预测
            batch_preds = []
            for i, logits in enumerate(logits_list):
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                batch_preds.append(preds)
            
            # 获取真实标签
            batch_targets = []
            for i, labels in enumerate(labels_list):
                targets = torch.argmax(labels, dim=1).cpu().numpy()
                batch_targets.append(targets)
            
            # 收集结果
            for j in range(len(image_names)):
                if samples_collected >= num_samples:
                    break
                
                results.append({
                    'image_name': image_names[j],
                    'image': images[j].cpu(),
                    'predictions': [batch_preds[i][j] for i in range(len(task_names))],
                    'targets': [batch_targets[i][j] for i in range(len(task_names))],
                    'masks': [masks[j][i].item() for i in range(len(task_names))]
                })
                samples_collected += 1
    
    # 创建可视化图像
    for idx, result in enumerate(results):
        # 反归一化图像
        img_tensor = result['image']
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img_tensor = img_tensor * std + mean
        img_tensor = torch.clamp(img_tensor, 0, 1)
        
        # 转换为PIL图像
        img_array = (img_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        img = Image.fromarray(img_array)
        
        # 创建画布
        canvas_width = 800
        canvas_height = max(400, 100 + len(task_names) * 25)
        canvas = Image.new('RGB', (canvas_width, canvas_height), (255, 255, 255))
        
        # 粘贴图像
        img_resized = img.resize((300, 300))
        canvas.paste(img_resized, (20, 50))
        
        # 添加文字
        draw = ImageDraw.Draw(canvas)
        
        # 尝试加载字体
        try:
            font_title = ImageFont.truetype("simhei.ttf", 20)
            font_text = ImageFont.truetype("simhei.ttf", 14)
        except:
            font_title = ImageFont.load_default()
            font_text = ImageFont.load_default()
        
        # 标题
        draw.text((350, 20), result['image_name'], fill=(0, 0, 0), font=font_title)
        
        # 任务标签
        y_offset = 60
        for i, task_name in enumerate(task_names):
            if result['masks'][i] > 0:
                pred = result['predictions'][i]
                target = result['targets'][i]
                class_names = TASK_CONFIG[task_name]['classes']
                pred_str = class_names[pred] if pred < len(class_names) else str(pred)
                target_str = class_names[target] if target < len(class_names) else str(target)
                
                correct = pred == target
                color = (0, 128, 0) if correct else (255, 0, 0)
                
                text = f"{task_name}: 真实={target_str}, 预测={pred_str}"
                draw.text((350, y_offset), text, fill=color, font=font_text)
            else:
                text = f"{task_name}: 缺失"
                draw.text((350, y_offset), text, fill=(128, 128, 128), font=font_text)
            
            y_offset += 22
        
        # 保存
        canvas.save(os.path.join(vis_dir, f'sample_{idx:03d}.png'))


def print_metrics_table(metrics, task_names):
    """打印指标表格"""
    print("\n" + "=" * 100)
    print(f"{'Task':<15} {'Accuracy':>12} {'Precision':>12} {'Recall':>12} {'F1-Score':>12} {'Support':>10}")
    print("-" * 100)
    
    for task_name in task_names:
        m = metrics[task_name]
        print(f"{task_name:<15} {m['accuracy']:>12.4f} {m['precision']:>12.4f} "
              f"{m['recall']:>12.4f} {m['f1']:>12.4f} {m['support']:>10}")
    
    print("-" * 100)
    print(f"{'Overall':<15} {metrics['overall']['accuracy']:>12.4f} {'-':>12} "
          f"{'-':>12} {metrics['overall']['f1']:>12.4f} {'-':>10}")
    print("=" * 100)


def main(args):
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 任务配置
    task_configs = [TASK_CONFIG[name]['num_classes'] for name in TASK_NAMES]
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载模型
    print(f"Loading model from {args.checkpoint}")
    model = load_model(args.checkpoint, task_configs, device)
    
    # 数据变换
    transform = get_transforms(train=False, image_size=args.image_size)
    
    # 加载数据集
    dataset = TongueDataset(
        csv_path=args.csv_path,
        image_dir=args.image_dir,
        transform=transform
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # 评估
    print("\nEvaluating...")
    all_predictions, all_targets, all_image_names = evaluate_model(
        model, dataloader, device, TASK_NAMES
    )
    
    # 计算指标
    metrics = compute_metrics(all_predictions, all_targets, TASK_NAMES)
    
    # 打印指标
    print_metrics_table(metrics, TASK_NAMES)
    
    # 保存混淆矩阵
    print("\nSaving confusion matrices...")
    save_confusion_matrices(all_predictions, all_targets, TASK_NAMES, args.output_dir)
    
    # 保存预测结果
    print("Saving predictions...")
    save_predictions_csv(all_predictions, all_targets, all_image_names, TASK_NAMES, args.output_dir)
    
    # 可视化
    if args.visualize:
        print("Creating visualizations...")
        visualize_predictions(model, dataloader, device, TASK_NAMES, args.output_dir, args.num_vis)
    
    # 保存指标
    metrics_df = pd.DataFrame([
        {
            'Task': task_name,
            'Accuracy': metrics[task_name]['accuracy'],
            'Precision': metrics[task_name]['precision'],
            'Recall': metrics[task_name]['recall'],
            'F1-Score': metrics[task_name]['f1'],
            'Support': metrics[task_name]['support']
        }
        for task_name in TASK_NAMES
    ])
    metrics_df.to_csv(os.path.join(args.output_dir, 'metrics.csv'), index=False, encoding='utf-8-sig')
    
    # 保存总体指标
    overall_df = pd.DataFrame([{
        'Overall_Accuracy': metrics['overall']['accuracy'],
        'Overall_F1': metrics['overall']['f1']
    }])
    overall_df.to_csv(os.path.join(args.output_dir, 'overall_metrics.csv'), index=False, encoding='utf-8-sig')
    
    print(f"\nResults saved to {args.output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate Tongue Classification Model')
    
    parser.add_argument('--checkpoint', type=str, default='outputs/best_model.pth', help='模型检查点路径')
    parser.add_argument('--csv_path', type=str, default='result.csv', help='CSV文件路径')
    parser.add_argument('--image_dir', type=str, default='images', help='图像目录')
    parser.add_argument('--output_dir', type=str, default='evaluation_results', help='输出目录')
    
    parser.add_argument('--image_size', type=int, default=224, help='图像大小')
    parser.add_argument('--batch_size', type=int, default=16, help='批次大小')
    parser.add_argument('--num_workers', type=int, default=4, help='数据加载线程数')
    parser.add_argument('--visualize', action='store_true', help='是否生成可视化结果')
    parser.add_argument('--num_vis', type=int, default=20, help='可视化样本数量')
    
    args = parser.parse_args()
    main(args)
