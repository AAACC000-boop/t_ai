import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from collections import Counter
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from dataset import TongueDataset, get_transforms, collate_fn, TASK_CONFIG, TASK_NAMES, read_csv_with_encoding
from model import create_model


def analyze_label_distribution(csv_path, output_dir):
    """分析数据集标签分布"""
    print("\nAnalyzing label distribution...")
    
    df = read_csv_with_encoding(csv_path)
    label_cols = df.columns[1:16]
    
    # 创建输出目录
    dist_dir = os.path.join(output_dir, 'label_distribution')
    os.makedirs(dist_dir, exist_ok=True)
    
    distribution_data = []
    
    for i, col in enumerate(label_cols):
        task_name = TASK_NAMES[i]
        num_classes = TASK_CONFIG[task_name]['num_classes']
        class_names = TASK_CONFIG[task_name]['classes']
        
        # 统计每个类别的数量
        values = df[col].values
        
        # 过滤缺失值
        valid_values = values[values != -1]
        valid_values = valid_values[~pd.isna(valid_values)]
        
        # 统计
        counter = Counter(valid_values.astype(int))
        
        # 保存数据
        for class_idx in range(num_classes):
            count = counter.get(class_idx, 0)
            distribution_data.append({
                'Task': task_name,
                'Class_Index': class_idx,
                'Class_Name': class_names[class_idx],
                'Count': count,
                'Percentage': count / len(valid_values) * 100 if len(valid_values) > 0 else 0
            })
        
        # 缺失值统计
        missing_count = len(values) - len(valid_values)
        distribution_data.append({
            'Task': task_name,
            'Class_Index': -1,
            'Class_Name': '缺失',
            'Count': missing_count,
            'Percentage': missing_count / len(values) * 100
        })
        
        # 绘制分布图
        fig, ax = plt.subplots(figsize=(10, 6))
        
        class_counts = [counter.get(i, 0) for i in range(num_classes)]
        bars = ax.bar(class_names, class_counts, color='steelblue')
        
        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}',
                   ha='center', va='bottom')
        
        ax.set_xlabel('Class')
        ax.set_ylabel('Count')
        ax.set_title(f'{task_name} Distribution (Valid: {len(valid_values)}, Missing: {missing_count})')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(dist_dir, f'{task_name}_distribution.png'), dpi=150)
        plt.close()
    
    # 保存分布数据
    dist_df = pd.DataFrame(distribution_data)
    dist_df.to_csv(os.path.join(dist_dir, 'label_distribution.csv'), index=False, encoding='utf-8-sig')
    
    print(f"Label distribution saved to {dist_dir}")


def visualize_tongue_color_samples(csv_path, image_dir, output_dir, num_samples_per_class=5):
    """可视化舌体颜色样本"""
    print("\nVisualizing tongue color samples...")
    
    df = read_csv_with_encoding(csv_path)
    image_col = df.columns[0]
    color_col = df.columns[1]  # 舌体颜色是第一列
    
    task_name = '舌体颜色'
    class_names = TASK_CONFIG[task_name]['classes']
    num_classes = len(class_names)
    
    # 创建输出目录
    vis_dir = os.path.join(output_dir, 'tongue_color_samples')
    os.makedirs(vis_dir, exist_ok=True)
    
    # 为每个类别收集样本
    fig, axes = plt.subplots(num_classes, num_samples_per_class, figsize=(num_samples_per_class * 3, num_classes * 3))
    
    for class_idx in range(num_classes):
        # 获取该类别的样本
        class_samples = df[df[color_col] == class_idx]
        
        if len(class_samples) == 0:
            continue
        
        # 随机选择样本
        samples = class_samples.sample(min(num_samples_per_class, len(class_samples)))
        
        for i, (_, row) in enumerate(samples.iterrows()):
            if i >= num_samples_per_class:
                break
            
            image_name = row[image_col]
            image_path = os.path.join(image_dir, image_name)
            
            try:
                img = Image.open(image_path).convert('RGB')
                
                ax = axes[class_idx, i] if num_classes > 1 else axes[i]
                ax.imshow(img)
                ax.axis('off')
                
                if i == 0:
                    ax.set_ylabel(class_names[class_idx], fontsize=12, rotation=0, ha='right', va='center')
            except Exception as e:
                print(f"Error loading {image_path}: {e}")
    
    plt.suptitle('Tongue Color Samples', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, 'tongue_color_samples.png'), dpi=150)
    plt.close()
    
    print(f"Tongue color samples saved to {vis_dir}")


def extract_features(model, dataloader, device):
    """提取特征向量"""
    print("\nExtracting features...")
    
    model.eval()
    features_list = []
    labels_list = {name: [] for name in TASK_NAMES}
    
    with torch.no_grad():
        for images, labels, masks, _ in tqdm(dataloader, desc='Extracting'):
            images = images.to(device)
            
            # 提取特征
            features = model.get_features(images)
            features_list.append(features.cpu().numpy())
            
            # 收集标签
            for i, task_name in enumerate(TASK_NAMES):
                task_labels = torch.argmax(labels[i], dim=1).cpu().numpy()
                task_masks = masks[:, i].cpu().numpy()
                
                for j in range(len(task_masks)):
                    labels_list[task_name].append({
                        'label': task_labels[j],
                        'valid': task_masks[j] > 0
                    })
    
    features = np.concatenate(features_list, axis=0)
    
    return features, labels_list


def visualize_pca_tsne(features, labels_list, output_dir, task_name='舌体颜色', max_samples=1000):
    """PCA和t-SNE可视化"""
    print(f"\nVisualizing PCA and t-SNE for {task_name}...")
    
    vis_dir = os.path.join(output_dir, 'feature_visualization')
    os.makedirs(vis_dir, exist_ok=True)
    
    # 获取该任务的标签
    task_labels_info = labels_list[task_name]
    
    # 过滤有效标签
    valid_indices = [i for i, info in enumerate(task_labels_info) if info['valid']]
    
    if len(valid_indices) == 0:
        print(f"No valid labels for {task_name}")
        return
    
    # 限制样本数量
    if len(valid_indices) > max_samples:
        valid_indices = np.random.choice(valid_indices, max_samples, replace=False)
    
    valid_features = features[valid_indices]
    valid_labels = [task_labels_info[i]['label'] for i in valid_indices]
    
    class_names = TASK_CONFIG[task_name]['classes']
    num_classes = len(class_names)
    
    # PCA
    print("Computing PCA...")
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(valid_features)
    
    # t-SNE
    print("Computing t-SNE...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(valid_indices) - 1))
    tsne_result = tsne.fit_transform(valid_features)
    
    # 绘制PCA
    plt.figure(figsize=(10, 8))
    colors = plt.cm.tab10(np.linspace(0, 1, num_classes))
    
    for i in range(num_classes):
        mask = np.array(valid_labels) == i
        if mask.sum() > 0:
            plt.scatter(pca_result[mask, 0], pca_result[mask, 1],
                       c=[colors[i]], label=class_names[i], alpha=0.6, s=30)
    
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
    plt.title(f'{task_name} - PCA Visualization')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, f'{task_name}_pca.png'), dpi=150)
    plt.close()
    
    # 绘制t-SNE
    plt.figure(figsize=(10, 8))
    
    for i in range(num_classes):
        mask = np.array(valid_labels) == i
        if mask.sum() > 0:
            plt.scatter(tsne_result[mask, 0], tsne_result[mask, 1],
                       c=[colors[i]], label=class_names[i], alpha=0.6, s=30)
    
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.title(f'{task_name} - t-SNE Visualization')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, f'{task_name}_tsne.png'), dpi=150)
    plt.close()
    
    print(f"Feature visualization saved to {vis_dir}")


def analyze_tongue_color_confusion(csv_path, predictions_csv, image_dir, output_dir):
    """分析舌体颜色混淆矩阵和错分样本"""
    print("\nAnalyzing tongue color confusion...")
    
    # 读取真实标签和预测结果
    df_true = read_csv_with_encoding(csv_path)
    df_pred = read_csv_with_encoding(predictions_csv)
    
    task_name = '舌体颜色'
    class_names = TASK_CONFIG[task_name]['classes']
    
    # 获取舌体颜色的真实值和预测值
    true_col = df_true.columns[1]
    pred_col = f'{task_name}_pred'
    
    true_labels = df_true[true_col].values
    pred_labels = df_pred[pred_col].values
    image_names = df_true[df_true.columns[0]].values
    
    # 过滤有效标签
    valid_mask = (true_labels != -1) & (~pd.isna(true_labels)) & (pred_labels != -1)
    true_labels = true_labels[valid_mask].astype(int)
    pred_labels = pred_labels[valid_mask].astype(int)
    image_names = image_names[valid_mask]
    
    # 创建输出目录
    analysis_dir = os.path.join(output_dir, 'tongue_color_analysis')
    os.makedirs(analysis_dir, exist_ok=True)
    
    # 计算混淆矩阵
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(true_labels, pred_labels, labels=list(range(len(class_names))))
    
    # 保存混淆矩阵
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    cm_df.to_csv(os.path.join(analysis_dir, 'confusion_matrix.csv'), encoding='utf-8-sig')
    
    # 绘制混淆矩阵热力图
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
               xticklabels=class_names, yticklabels=class_names)
    plt.title(f'{task_name} - Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(analysis_dir, 'confusion_matrix.png'), dpi=150)
    plt.close()
    
    # 找出错分样本
    misclassified = []
    for i in range(len(true_labels)):
        if true_labels[i] != pred_labels[i]:
            misclassified.append({
                'image_name': image_names[i],
                'true_label': class_names[true_labels[i]],
                'pred_label': class_names[pred_labels[i]],
                'true_idx': true_labels[i],
                'pred_idx': pred_labels[i]
            })
    
    # 保存错分样本信息
    misclassified_df = pd.DataFrame(misclassified)
    misclassified_df.to_csv(os.path.join(analysis_dir, 'misclassified_samples.csv'), 
                           index=False, encoding='utf-8-sig')
    
    # 可视化部分错分样本
    num_vis = min(20, len(misclassified))
    if num_vis > 0:
        fig, axes = plt.subplots(4, 5, figsize=(15, 12))
        axes = axes.flatten()
        
        for i in range(num_vis):
            info = misclassified[i]
            image_path = os.path.join(image_dir, info['image_name'])
            
            try:
                img = Image.open(image_path).convert('RGB')
                axes[i].imshow(img)
                axes[i].axis('off')
                axes[i].set_title(f"True: {info['true_label']}\nPred: {info['pred_label']}", 
                                 fontsize=10, color='red')
            except Exception as e:
                print(f"Error loading {image_path}: {e}")
        
        # 隐藏多余的子图
        for i in range(num_vis, 20):
            axes[i].axis('off')
        
        plt.suptitle('Misclassified Tongue Color Samples', fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(analysis_dir, 'misclassified_samples.png'), dpi=150)
        plt.close()
    
    print(f"Tongue color analysis saved to {analysis_dir}")
    print(f"Total misclassified: {len(misclassified)} / {len(true_labels)} "
          f"({len(misclassified)/len(true_labels)*100:.2f}%)")


def main(args):
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 1. 标签分布分析
    analyze_label_distribution(args.csv_path, args.output_dir)
    
    # 2. 舌体颜色样本可视化
    visualize_tongue_color_samples(args.csv_path, args.image_dir, args.output_dir, 
                                   args.num_color_samples)
    
    # 3. 特征提取和可视化（需要模型）
    if args.checkpoint and os.path.exists(args.checkpoint):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"\nUsing device: {device}")
        
        # 任务配置
        task_configs = [TASK_CONFIG[name]['num_classes'] for name in TASK_NAMES]
        
        # 加载模型
        print(f"Loading model from {args.checkpoint}")
        model = create_model(task_configs=task_configs, pretrained=False)
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        model.eval()
        
        # 数据加载
        transform = get_transforms(train=False, image_size=args.image_size)
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
        
        # 提取特征
        features, labels_list = extract_features(model, dataloader, device)
        
        # PCA和t-SNE可视化
        visualize_pca_tsne(features, labels_list, args.output_dir, 
                          task_name='舌体颜色', max_samples=args.max_tsne_samples)
    else:
        print(f"\nSkipping feature visualization (checkpoint not found: {args.checkpoint})")
    
    # 4. 舌体颜色混淆矩阵分析（需要预测结果）
    if args.predictions_csv and os.path.exists(args.predictions_csv):
        analyze_tongue_color_confusion(args.csv_path, args.predictions_csv, 
                                      args.image_dir, args.output_dir)
    else:
        print(f"\nSkipping confusion analysis (predictions not found: {args.predictions_csv})")
    
    print(f"\nAnalysis completed! Results saved to {args.output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analyze Tongue Dataset')
    
    parser.add_argument('--csv_path', type=str, default='result.csv', help='CSV文件路径')
    parser.add_argument('--image_dir', type=str, default='images', help='图像目录')
    parser.add_argument('--output_dir', type=str, default='analysis_results', help='输出目录')
    
    parser.add_argument('--checkpoint', type=str, default='outputs/best_model.pth', 
                       help='模型检查点路径（用于特征提取）')
    parser.add_argument('--predictions_csv', type=str, default='evaluation_results/predictions.csv',
                       help='预测结果CSV路径（用于混淆矩阵分析）')
    
    parser.add_argument('--image_size', type=int, default=224, help='图像大小')
    parser.add_argument('--batch_size', type=int, default=16, help='批次大小')
    parser.add_argument('--num_workers', type=int, default=4, help='数据加载线程数')
    parser.add_argument('--num_color_samples', type=int, default=5, help='每类舌体颜色样本数')
    parser.add_argument('--max_tsne_samples', type=int, default=1000, help='t-SNE最大样本数')
    
    args = parser.parse_args()
    main(args)
