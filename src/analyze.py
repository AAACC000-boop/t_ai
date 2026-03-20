# -*- coding: utf-8 -*-
"""
数据分析脚本
"""
import os
import sys
import csv
from typing import Dict, List, Any, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# 添加src路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import load_config, set_seed, get_device, load_checkpoint
from src.model import create_model
from src.dataset import create_test_dataloader


def analyze_label_distribution(
    csv_file: str,
    config: Dict[str, Any],
    output_dir: str
) -> None:
    """
    分析数据集标签分布
    """
    print("分析标签分布...")
    
    task_names = config["task"]["task_names"]
    class_names = config["task"]["class_names"]
    num_classes_list = config["task"]["num_classes"]
    
    # 读取数据（尝试不同编码）
    try:
        df = pd.read_csv(csv_file, encoding="utf-8")
    except:
        try:
            df = pd.read_csv(csv_file, encoding="gbk")
        except:
            df = pd.read_csv(csv_file, encoding="gb2312")
    
    # 创建输出目录
    dist_dir = os.path.join(output_dir, "label_distribution")
    os.makedirs(dist_dir, exist_ok=True)
    
    distribution_stats = []
    
    # 分析每个任务的标签分布
    for task_idx in range(15):
        task_name = task_names[task_idx]
        labels = df.iloc[:, task_idx + 1].values
        
        # 统计各标签数量
        label_counts = {}
        for label in labels:
            try:
                label = int(label)
                if label < 0:
                    label = -1
            except:
                label = -1
            label_counts[label] = label_counts.get(label, 0) + 1
        
        # 计算缺失率
        total = len(labels)
        missing = label_counts.get(-1, 0)
        missing_rate = missing / total
        
        # 准备统计信息
        stats = {
            "任务": task_name,
            "总样本数": total,
            "缺失数": missing,
            "缺失率": f"{missing_rate:.4f}"
        }
        
        # 添加各类别的数量
        for cls_idx in range(num_classes_list[task_idx]):
            cls_name = class_names[task_idx][cls_idx]
            count = label_counts.get(cls_idx, 0)
            stats[cls_name] = count
        
        distribution_stats.append(stats)
        
        # 绘制分布图
        plt.figure(figsize=(10, 6))
        counts = [label_counts.get(i, 0) for i in range(num_classes_list[task_idx])]
        bars = plt.bar(class_names[task_idx], counts)
        
        # 在柱子上添加数值标签
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                     f'{int(height)}', ha='center', va='bottom')
        
        plt.title(f"标签分布 - {task_name}\n(缺失率: {missing_rate:.4f})")
        plt.xlabel("类别")
        plt.ylabel("样本数")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(dist_dir, f"distribution_{task_name}.png"), dpi=300, bbox_inches="tight")
        plt.close()
    
    # 保存统计结果到CSV
    stats_path = os.path.join(dist_dir, "distribution_stats.csv")
    with open(stats_path, "w", newline="", encoding="utf-8") as f:
        if distribution_stats:
            writer = csv.DictWriter(f, fieldnames=distribution_stats[0].keys())
            writer.writeheader()
            writer.writerows(distribution_stats)
    
    print(f"标签分布分析结果已保存到: {dist_dir}")


def visualize_tongue_color_samples(
    image_dir: str,
    csv_file: str,
    config: Dict[str, Any],
    output_dir: str,
    samples_per_class: int = 5
) -> None:
    """
    舌体颜色样本可视化
    """
    print("可视化舌体颜色样本...")
    
    from PIL import Image
    
    color_classes = config["task"]["class_names"][0]  # 舌体颜色类别
    output_subdir = os.path.join(output_dir, "tongue_color_samples")
    os.makedirs(output_subdir, exist_ok=True)
    
    # 读取数据
    df = pd.read_csv(csv_file, encoding="utf-8")
    
    # 按颜色类别分组
    for color_idx, color_name in enumerate(color_classes):
        # 获取该类别的样本
        mask = (df.iloc[:, 1] == color_idx)  # 第2列是舌体颜色标签
        samples = df[mask].iloc[:, 0].head(samples_per_class).tolist()
        
        if not samples:
            print(f"类别 {color_name} 没有样本")
            continue
        
        # 创建子图
        fig, axes = plt.subplots(1, min(samples_per_class, len(samples)), 
                                 figsize=(3 * min(samples_per_class, len(samples)), 4))
        if len(samples) == 1:
            axes = [axes]
        
        for i, (img_name, ax) in enumerate(zip(samples, axes)):
            img_path = os.path.join(image_dir, img_name)
            try:
                img = Image.open(img_path).convert("RGB")
                ax.imshow(img)
                ax.set_title(f"{color_name} - {i+1}")
                ax.axis("off")
            except Exception as e:
                print(f"无法读取图像 {img_path}: {e}")
                ax.axis("off")
        
        plt.suptitle(f"舌体颜色: {color_name}", fontsize=16, y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(output_subdir, f"color_{color_name}.png"), 
                    dpi=300, bbox_inches="tight")
        plt.close()
    
    print(f"舌体颜色样本可视化结果已保存到: {output_subdir}")


def extract_features(
    model,
    dataloader,
    device
) -> Tuple[np.ndarray, List[str], List[List[int]]]:
    """
    提取特征向量
    """
    model.eval()
    
    all_features = []
    all_filenames = []
    all_labels = []
    
    with torch.no_grad():
        for images, targets_list, filenames in tqdm(dataloader, desc="提取特征"):
            images = images.to(device)
            
            # 提取特征
            features = model.get_features(images)
            features = features.cpu().numpy()
            
            all_features.append(features)
            all_filenames.extend(filenames)
            
            # 收集标签（舌体颜色）
            for targets in targets_list:
                if targets.dim() > 1:
                    valid_mask = targets.sum(dim=1) > 0
                    labels = torch.where(valid_mask, targets.argmax(dim=1), -1)
                else:
                    labels = targets
                all_labels.extend(labels.cpu().numpy().tolist())
                break  # 只取舌体颜色标签
    
    all_features = np.concatenate(all_features, axis=0)
    
    return all_features, all_filenames, all_labels


def visualize_pca_tsne(
    features: np.ndarray,
    labels: List[int],
    config: Dict[str, Any],
    output_dir: str
) -> None:
    """
    PCA + t-SNE 可视化
    """
    print("进行PCA和t-SNE可视化...")
    
    color_classes = config["task"]["class_names"][0]
    output_subdir = os.path.join(output_dir, "pca_tsne")
    os.makedirs(output_subdir, exist_ok=True)
    
    # 过滤掉缺失标签的样本
    valid_indices = [i for i, label in enumerate(labels) if label != -1]
    valid_features = features[valid_indices]
    valid_labels = [labels[i] for i in valid_indices]
    
    if len(valid_features) < 2:
        print("有效样本太少，无法进行可视化")
        return
    
    # 标准化特征
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(valid_features)
    
    # PCA可视化
    print("计算PCA...")
    pca = PCA(n_components=2, random_state=42)
    pca_result = pca.fit_transform(features_scaled)
    
    plt.figure(figsize=(12, 10))
    for color_idx, color_name in enumerate(color_classes):
        mask = [label == color_idx for label in valid_labels]
        if any(mask):
            plt.scatter(
                pca_result[mask, 0], pca_result[mask, 1],
                label=color_name, alpha=0.7, s=50
            )
    
    plt.title(f"PCA可视化 (舌体颜色)\n解释方差比: {pca.explained_variance_ratio_[:2].sum():.4f}")
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.4f})")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.4f})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_subdir, "pca_visualization.png"), dpi=300)
    plt.close()
    
    # t-SNE可视化
    print("计算t-SNE...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    tsne_result = tsne.fit_transform(features_scaled)
    
    plt.figure(figsize=(12, 10))
    for color_idx, color_name in enumerate(color_classes):
        mask = [label == color_idx for label in valid_labels]
        if any(mask):
            plt.scatter(
                tsne_result[mask, 0], tsne_result[mask, 1],
                label=color_name, alpha=0.7, s=50
            )
    
    plt.title("t-SNE可视化 (舌体颜色)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_subdir, "tsne_visualization.png"), dpi=300)
    plt.close()
    
    # 保存PCA解释方差
    pca_full = PCA(random_state=42)
    pca_full.fit(features_scaled)
    
    plt.figure(figsize=(10, 6))
    cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)
    plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, "bo-")
    plt.xlabel("主成分数量")
    plt.ylabel("累积解释方差比")
    plt.title("PCA累积解释方差比")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_subdir, "pca_explained_variance.png"), dpi=300)
    plt.close()
    
    print(f"PCA/t-SNE可视化结果已保存到: {output_subdir}")


def analyze_color_confusion(
    predictions: List[List[int]],
    ground_truth: List[List[int]],
    filenames: List[str],
    config: Dict[str, Any],
    output_dir: str
) -> None:
    """
    舌体颜色混淆矩阵和错分样本导出
    """
    print("分析舌体颜色混淆情况...")
    
    color_classes = config["task"]["class_names"][0]
    output_subdir = os.path.join(output_dir, "color_confusion")
    os.makedirs(output_subdir, exist_ok=True)
    
    from PIL import Image
    
    # 提取舌体颜色的预测和真实标签（任务0）
    color_preds = [p[0] for p in predictions]
    color_gts = [g[0] for g in ground_truth]
    
    # 过滤有效样本
    valid_data = [(p, g, f) for p, g, f in zip(color_preds, color_gts, filenames) if g != -1]
    
    if not valid_data:
        print("没有有效的舌体颜色标签")
        return
    
    valid_preds, valid_gts, valid_files = zip(*valid_data)
    
    # 计算混淆矩阵
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(valid_gts, valid_preds)
    
    # 绘制混淆矩阵
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=color_classes, yticklabels=color_classes
    )
    plt.title("舌体颜色混淆矩阵")
    plt.ylabel("真实标签")
    plt.xlabel("预测标签")
    plt.tight_layout()
    plt.savefig(os.path.join(output_subdir, "color_confusion_matrix.png"), dpi=300)
    plt.close()
    
    # 保存混淆矩阵为CSV
    cm_path = os.path.join(output_subdir, "color_confusion_matrix.csv")
    with open(cm_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["真实\预测"] + color_classes)
        for i, row in enumerate(cm):
            writer.writerow([color_classes[i]] + row.tolist())
    
    # 导出错分样本
    misclassified_dir = os.path.join(output_subdir, "misclassified_samples")
    os.makedirs(misclassified_dir, exist_ok=True)
    
    misclassified = [(p, g, f) for p, g, f in valid_data if p != g]
    
    if misclassified:
        print(f"导出 {len(misclassified)} 个错分样本...")
        
        # 按真实类别分组
        for true_idx, true_color in enumerate(color_classes):
            group = [(p, f) for p, g, f in misclassified if g == true_idx]
            if not group:
                continue
            
            # 保存到CSV
            group_path = os.path.join(misclassified_dir, f"true_{true_color}.csv")
            with open(group_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["文件名", "真实标签", "预测标签"])
                for pred, fname in group:
                    writer.writerow([fname, true_color, color_classes[pred]])
            
            # 可视化前几个样本
            n_samples = min(10, len(group))
            fig, axes = plt.subplots(2, (n_samples + 1) // 2, figsize=(15, 8))
            axes = axes.flatten()
            
            for i, (pred, fname) in enumerate(group[:n_samples]):
                img_path = os.path.join(config["data"]["image_dir"], fname)
                try:
                    img = Image.open(img_path).convert("RGB")
                    axes[i].imshow(img)
                    axes[i].set_title(f"真实: {true_color}\n预测: {color_classes[pred]}", fontsize=10)
                    axes[i].axis("off")
                except:
                    axes[i].axis("off")
            
            # 隐藏多余的子图
            for i in range(n_samples, len(axes)):
                axes[i].axis("off")
            
            plt.suptitle(f"舌体颜色错分样本 - 真实类别: {true_color}", fontsize=14)
            plt.tight_layout()
            plt.savefig(os.path.join(misclassified_dir, f"misclassified_true_{true_color}.png"), 
                        dpi=300, bbox_inches="tight")
            plt.close()
    
    print(f"舌体颜色混淆分析结果已保存到: {output_subdir}")


def main_analysis(
    checkpoint_path = None,
    csv_file = None,
    output_dir = None
):
    """
    主分析函数
    """
    # 加载配置
    config = load_config()
    
    if output_dir is None:
        output_dir = config["paths"]["analysis_dir"]
    os.makedirs(output_dir, exist_ok=True)
    
    if csv_file is None:
        csv_file = config["data"]["csv_file"]
    
    # 设置随机种子
    set_seed(config["data"]["random_seed"])
    
    print("=" * 80)
    print("开始数据分析")
    print("=" * 80)
    
    # 1. 标签分布分析
    print("\n[1/4] 标签分布分析")
    analyze_label_distribution(csv_file, config, output_dir)
    
    # 2. 舌体颜色样本可视化
    print("\n[2/4] 舌体颜色样本可视化")
    visualize_tongue_color_samples(
        config["data"]["image_dir"],
        csv_file,
        config,
        output_dir
    )
    
    # 如果提供了模型检查点，则进行特征提取和混淆分析
    if checkpoint_path and os.path.exists(checkpoint_path):
        device = get_device()
        print(f"\n使用设备: {device}")
        
        # 创建模型
        print("创建模型...")
        model = create_model(config)
        model = model.to(device)
        
        # 加载检查点
        print(f"加载检查点: {checkpoint_path}")
        load_checkpoint(checkpoint_path, model, device=device)
        
        # 创建数据加载器
        print("创建数据加载器...")
        dataloader = create_test_dataloader(config, csv_file)
        
        # 3. 特征提取和PCA/t-SNE可视化
        print("\n[3/4] 特征提取和PCA/t-SNE可视化")
        features, filenames, labels = extract_features(model, dataloader, device)
        visualize_pca_tsne(features, labels, config, output_dir)
        
        # 4. 舌体颜色混淆分析
        print("\n[4/4] 舌体颜色混淆矩阵和错分样本导出")
        from src.infer import inference, load_ground_truth, align_predictions
        
        # 推理
        print("运行推理...")
        pred_filenames, predictions = inference(model, dataloader, device)
        
        # 加载真实标签
        gt_filenames, ground_truth = load_ground_truth(csv_file, config)
        
        # 对齐
        aligned_preds, aligned_gts = align_predictions(pred_filenames, predictions, gt_filenames, ground_truth)
        
        # 混淆分析
        analyze_color_confusion(aligned_preds, aligned_gts, pred_filenames, config, output_dir)
    else:
        print("\n[3/4] 跳过特征提取可视化（未提供模型检查点）")
        print("\n[4/4] 跳过混淆分析（未提供模型检查点）")
    
    print("\n" + "=" * 80)
    print("数据分析完成!")
    print("=" * 80)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="舌象识别数据分析")
    parser.add_argument("--checkpoint", type=str, help="模型检查点路径（可选，用于特征可视化和混淆分析）")
    parser.add_argument("--csv_file", type=str, help="CSV文件路径")
    parser.add_argument("--output_dir", type=str, help="输出目录")
    
    args = parser.parse_args()
    
    main_analysis(args.checkpoint, args.csv_file, args.output_dir)
