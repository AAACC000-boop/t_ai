# -*- coding: utf-8 -*-
"""
推理和评估脚本
"""
import os
import sys
import csv
from typing import Dict, List, Any, Tuple, Optional

import torch
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
)
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import seaborn as sns

# 添加src路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import (
    load_config, set_seed, get_device, create_dirs, load_checkpoint
)
from src.model import create_model
from src.dataset import create_test_dataloader, create_inference_dataloader


def inference(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device
) -> Tuple[List[str], List[List[int]]]:
    """
    模型推理
    返回: 文件名列表, 15个任务的预测结果列表
    """
    model.eval()
    
    all_filenames = []
    all_predictions = [[] for _ in range(15)]  # 15个任务
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="推理"):
            # 处理两种返回格式: 
            # - 推理模式: (images, filenames)
            # - 评估模式: (images, targets_list, filenames)
            if len(batch) == 3:
                images, _, filenames = batch
            else:
                images, filenames = batch
            images = images.to(device)
            
            # 前向传播
            logits_list = model(images)
            
            # 收集预测结果
            for task_idx, logits in enumerate(logits_list):
                predictions = logits.argmax(dim=1).cpu().numpy()
                all_predictions[task_idx].extend(predictions.tolist())
            
            all_filenames.extend(filenames)
    
    # 转换为 [样本数, 任务数] 的格式
    predictions_by_sample = []
    for i in range(len(all_filenames)):
        sample_pred = [all_predictions[task_idx][i] for task_idx in range(15)]
        predictions_by_sample.append(sample_pred)
    
    return all_filenames, predictions_by_sample


def save_predictions(
    filenames: List[str],
    predictions: List[List[int]],
    output_path: str,
    config: Dict[str, Any]
) -> None:
    """
    保存预测结果到CSV
    """
    task_names = config["task"]["task_names"]
    
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        
        # 写入表头
        header = ["文件名"] + task_names
        writer.writerow(header)
        
        # 写入数据
        for filename, pred in zip(filenames, predictions):
            row = [filename] + pred
            writer.writerow(row)


def load_ground_truth(csv_file: str, config: Dict[str, Any]) -> Tuple[List[str], List[List[int]]]:
    """
    加载真实标签
    """
    # 读取CSV文件（尝试不同编码）
    try:
        df = pd.read_csv(csv_file, encoding="utf-8")
    except:
        try:
            df = pd.read_csv(csv_file, encoding="gbk")
        except:
            df = pd.read_csv(csv_file, encoding="gb2312")
    
    filenames = df.iloc[:, 0].tolist()
    ground_truth = []
    
    for _, row in df.iterrows():
        gt = []
        for i in range(1, 16):
            try:
                label = int(row[i])
                gt.append(label if label >= 0 else -1)
            except (ValueError, TypeError):
                gt.append(-1)
        ground_truth.append(gt)
    
    return filenames, ground_truth


def align_predictions(
    filenames: List[str],
    predictions: List[List[int]],
    gt_filenames: List[str],
    ground_truth: List[List[int]]
) -> Tuple[List[List[int]], List[List[int]]]:
    """
    对齐预测结果和真实标签
    """
    # 创建文件名到索引的映射
    gt_dict = {fname: gt for fname, gt in zip(gt_filenames, ground_truth)}
    
    aligned_preds = []
    aligned_gts = []
    
    for fname, pred in zip(filenames, predictions):
        if fname in gt_dict:
            aligned_preds.append(pred)
            aligned_gts.append(gt_dict[fname])
    
    return aligned_preds, aligned_gts


def calculate_metrics(
    predictions: List[List[int]],
    ground_truth: List[List[int]],
    config: Dict[str, Any]
) -> Dict[str, Dict[str, float]]:
    """
    计算每个任务的评估指标
    """
    num_tasks = config["task"]["num_tasks"]
    task_names = config["task"]["task_names"]
    
    metrics = {}
    all_f1 = []
    
    for task_idx in range(num_tasks):
        # 收集有效样本
        preds = []
        gts = []
        
        for p, g in zip(predictions, ground_truth):
            if g[task_idx] != -1:  # 忽略缺失标签
                preds.append(p[task_idx])
                gts.append(g[task_idx])
        
        if len(gts) == 0:
            metrics[task_names[task_idx]] = {
                "accuracy": 0.0,
                "f1": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "support": 0
            }
            continue
        
        # 计算指标
        accuracy = accuracy_score(gts, preds)
        f1 = f1_score(gts, preds, average="weighted", zero_division=0)
        precision = precision_score(gts, preds, average="weighted", zero_division=0)
        recall = recall_score(gts, preds, average="weighted", zero_division=0)
        
        metrics[task_names[task_idx]] = {
            "accuracy": float(accuracy),
            "f1": float(f1),
            "precision": float(precision),
            "recall": float(recall),
            "support": len(gts)
        }
        
        all_f1.append(f1)
    
    # 计算总体指标
    metrics["总体"] = {
        "accuracy": float(np.mean([m["accuracy"] for m in metrics.values()])),
        "f1": float(np.mean(all_f1)) if all_f1 else 0.0,
        "precision": float(np.mean([m["precision"] for m in metrics.values()])),
        "recall": float(np.mean([m["recall"] for m in metrics.values()])),
        "support": sum(m["support"] for m in metrics.values())
    }
    
    return metrics


def save_metrics(metrics: Dict[str, Dict[str, float]], output_path: str) -> None:
    """
    保存评估指标到CSV
    """
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        
        # 表头
        writer.writerow(["任务", "准确率", "F1", "精确率", "召回率", "样本数"])
        
        # 数据
        for task_name, m in metrics.items():
            writer.writerow([
                task_name,
                f"{m['accuracy']:.4f}",
                f"{m['f1']:.4f}",
                f"{m['precision']:.4f}",
                f"{m['recall']:.4f}",
                m["support"]
            ])


def save_confusion_matrices(
    predictions: List[List[int]],
    ground_truth: List[List[int]],
    config: Dict[str, Any],
    output_dir: str
) -> None:
    """
    保存每个任务的混淆矩阵
    """
    num_tasks = config["task"]["num_tasks"]
    task_names = config["task"]["task_names"]
    class_names = config["task"]["class_names"]
    
    os.makedirs(output_dir, exist_ok=True)
    
    for task_idx in range(num_tasks):
        # 收集有效样本
        preds = []
        gts = []
        
        for p, g in zip(predictions, ground_truth):
            if g[task_idx] != -1:
                preds.append(p[task_idx])
                gts.append(g[task_idx])
        
        if len(gts) == 0:
            continue
        
        # 计算混淆矩阵
        cm = confusion_matrix(gts, preds)
        classes = class_names[task_idx]
        
        # 保存为CSV
        cm_path = os.path.join(output_dir, f"confusion_matrix_{task_names[task_idx]}.csv")
        with open(cm_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["真实\预测"] + classes)
            for i, row in enumerate(cm):
                writer.writerow([classes[i]] + row.tolist())
        
        # 保存为图片
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=classes, yticklabels=classes
        )
        plt.title(f"混淆矩阵 - {task_names[task_idx]}")
        plt.ylabel("真实标签")
        plt.xlabel("预测标签")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"confusion_matrix_{task_names[task_idx]}.png"), dpi=300)
        plt.close()


def visualize_predictions(
    image_dir: str,
    filenames: List[str],
    predictions: List[List[int]],
    ground_truth: Optional[List[List[int]]],
    config: Dict[str, Any],
    output_dir: str,
    max_samples: int = 50
) -> None:
    """
    可视化预测结果
    """
    task_names = config["task"]["task_names"]
    class_names = config["task"]["class_names"]
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 尝试加载字体
    try:
        font = ImageFont.truetype("msyh.ttc", 12)
    except:
        font = ImageFont.load_default()
    
    for i, (filename, pred) in enumerate(tqdm(zip(filenames[:max_samples], predictions[:max_samples]), desc="可视化")):
        img_path = os.path.join(image_dir, filename)
        
        try:
            img = Image.open(img_path).convert("RGB")
        except:
            continue
        
        # 调整图像大小以便显示文字
        draw_img = img.resize((400, 400))
        draw = ImageDraw.Draw(draw_img)
        
        # 绘制预测结果
        y_offset = 10
        for task_idx, p in enumerate(pred):
            gt_text = ""
            if ground_truth is not None and ground_truth[i][task_idx] != -1:
                gt = ground_truth[i][task_idx]
                gt_text = f" GT: {class_names[task_idx][gt]}"
                color = "green" if p == gt else "red"
            else:
                color = "black"
            
            text = f"{task_names[task_idx]}: {class_names[task_idx][p]}{gt_text}"
            draw.text((10, y_offset), text, fill=color, font=font)
            y_offset += 20
        
        # 保存
        output_path = os.path.join(output_dir, f"viz_{i}_{filename}")
        draw_img.save(output_path)


def print_metrics_summary(metrics: Dict[str, Dict[str, float]]) -> None:
    """
    打印指标摘要
    """
    print("\n" + "=" * 80)
    print("评估指标摘要")
    print("=" * 80)
    print(f"{'任务':<12} {'准确率':<10} {'F1':<10} {'精确率':<10} {'召回率':<10} {'样本数'}")
    print("-" * 80)
    
    for task_name, m in metrics.items():
        print(f"{task_name:<12} {m['accuracy']:<10.4f} {m['f1']:<10.4f} "
              f"{m['precision']:<10.4f} {m['recall']:<10.4f} {m['support']}")
    
    print("=" * 80)


def main_inference(
    checkpoint_path: str,
    image_dir: Optional[str] = None,
    image_files: Optional[List[str]] = None,
    output_dir: Optional[str] = None
) -> None:
    """
    主推理函数（无标签）
    """
    # 加载配置
    config = load_config()
    
    if output_dir is None:
        output_dir = config["paths"]["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    
    # 设置随机种子
    set_seed(config["data"]["random_seed"])
    
    # 获取设备
    device = get_device()
    print(f"使用设备: {device}")
    
    # 创建模型
    print("创建模型...")
    model = create_model(config)
    model = model.to(device)
    
    # 加载检查点
    print(f"加载检查点: {checkpoint_path}")
    load_checkpoint(checkpoint_path, model, device=device)
    
    # 创建数据加载器
    print("创建数据加载器...")
    dataloader = create_inference_dataloader(config, image_dir, image_files)
    print(f"样本数: {len(dataloader.dataset)}")
    
    # 推理
    print("开始推理...")
    filenames, predictions = inference(model, dataloader, device)
    
    # 保存预测结果
    output_path = os.path.join(output_dir, "predictions.csv")
    save_predictions(filenames, predictions, output_path, config)
    print(f"预测结果已保存到: {output_path}")
    
    # 可视化
    print("生成可视化结果...")
    visualize_predictions(
        image_dir or config["data"]["image_dir"],
        filenames, predictions, None, config,
        os.path.join(output_dir, "visualization")
    )
    print("完成!")


def main_evaluate(
    checkpoint_path: str,
    csv_file: Optional[str] = None,
    output_dir: Optional[str] = None
) -> None:
    """
    主评估函数（有标签）
    """
    # 加载配置
    config = load_config()
    
    if output_dir is None:
        output_dir = config["paths"]["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    
    # 设置随机种子
    set_seed(config["data"]["random_seed"])
    
    # 获取设备
    device = get_device()
    print(f"使用设备: {device}")
    
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
    print(f"样本数: {len(dataloader.dataset)}")
    
    # 推理
    print("开始推理...")
    filenames, predictions = inference(model, dataloader, device)
    
    # 保存预测结果
    pred_output_path = os.path.join(output_dir, "predictions.csv")
    save_predictions(filenames, predictions, pred_output_path, config)
    print(f"预测结果已保存到: {pred_output_path}")
    
    # 加载真实标签
    print("加载真实标签...")
    if csv_file is None:
        csv_file = config["data"]["csv_file"]
    gt_filenames, ground_truth = load_ground_truth(csv_file, config)
    
    # 对齐预测和标签
    aligned_preds, aligned_gts = align_predictions(filenames, predictions, gt_filenames, ground_truth)
    print(f"有效样本数: {len(aligned_gts)}")
    
    # 计算指标
    print("计算评估指标...")
    metrics = calculate_metrics(aligned_preds, aligned_gts, config)
    
    # 保存指标
    metrics_output_path = os.path.join(output_dir, "evaluation_metrics.csv")
    save_metrics(metrics, metrics_output_path)
    print(f"评估指标已保存到: {metrics_output_path}")
    
    # 打印摘要
    print_metrics_summary(metrics)
    
    # 保存混淆矩阵
    print("生成混淆矩阵...")
    save_confusion_matrices(
        aligned_preds, aligned_gts, config,
        os.path.join(output_dir, "confusion_matrices")
    )
    
    # 可视化
    print("生成可视化结果...")
    visualize_predictions(
        config["data"]["image_dir"],
        filenames, predictions, ground_truth, config,
        os.path.join(output_dir, "visualization")
    )
    
    print("完成!")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="舌象识别推理和评估")
    parser.add_argument("--checkpoint", type=str, required=True, help="模型检查点路径")
    parser.add_argument("--mode", type=str, choices=["infer", "eval"], default="eval", help="运行模式")
    parser.add_argument("--image_dir", type=str, help="推理时的图像目录")
    parser.add_argument("--csv_file", type=str, help="评估时的CSV文件路径")
    parser.add_argument("--output_dir", type=str, help="输出目录")
    
    args = parser.parse_args()
    
    if args.mode == "infer":
        main_inference(args.checkpoint, args.image_dir, None, args.output_dir)
    else:
        main_evaluate(args.checkpoint, args.csv_file, args.output_dir)
