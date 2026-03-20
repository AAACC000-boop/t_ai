import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image, ImageDraw, ImageFont
from datetime import datetime
import json
import glob

import config
from dataset import get_full_dataset_loader
from model import load_model


def run_inference(model, data_loader, device):
    model.eval()
    
    all_predictions = [[] for _ in range(config.NUM_TASKS)]
    all_targets = [[] for _ in range(config.NUM_TASKS)]
    all_image_names = []
    
    with torch.no_grad():
        pbar = tqdm(data_loader, desc="Inference")
        for images, labels, img_names in pbar:
            images = images.to(device)
            
            outputs = model(images)
            
            for i, output in enumerate(outputs):
                all_predictions[i].append(output.cpu().numpy())
            
            for i, label in enumerate(labels):
                all_targets[i].append(label.numpy())
            
            all_image_names.extend(img_names)
    
    all_predictions = [np.concatenate(p, axis=0) for p in all_predictions]
    all_targets = [np.concatenate(t, axis=0) for t in all_targets]
    
    pred_labels = [np.argmax(p, axis=1) for p in all_predictions]
    target_labels = [np.argmax(t, axis=1) for t in all_targets]
    
    valid_masks = []
    for t in all_targets:
        valid_masks.append(t.sum(axis=1) > 0)
    
    return pred_labels, target_labels, valid_masks, all_image_names


def compute_all_metrics(pred_labels, target_labels, valid_masks):
    all_metrics = []
    
    for i, (pred, target, mask) in enumerate(zip(pred_labels, target_labels, valid_masks)):
        if mask.sum() == 0:
            all_metrics.append({
                'accuracy': 0.0,
                'f1': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'valid_samples': 0
            })
            continue
        
        pred_valid = pred[mask]
        target_valid = target[mask]
        
        metrics = {
            'accuracy': accuracy_score(target_valid, pred_valid),
            'f1': f1_score(target_valid, pred_valid, average='weighted', zero_division=0),
            'precision': precision_score(target_valid, pred_valid, average='weighted', zero_division=0),
            'recall': recall_score(target_valid, pred_valid, average='weighted', zero_division=0),
            'valid_samples': int(mask.sum())
        }
        all_metrics.append(metrics)
    
    return all_metrics


def save_confusion_matrices(pred_labels, target_labels, valid_masks, output_dir):
    cm_dir = output_dir / "confusion_matrices"
    cm_dir.mkdir(exist_ok=True)
    
    for i, (pred, target, mask) in enumerate(zip(pred_labels, target_labels, valid_masks)):
        task_def = config.TASK_DEFINITIONS[i]
        num_classes = task_def['num_classes']
        
        if mask.sum() == 0:
            continue
        
        pred_valid = pred[mask]
        target_valid = target[mask]
        
        cm = confusion_matrix(target_valid, pred_valid, labels=list(range(num_classes)))
        
        cm_df = pd.DataFrame(
            cm,
            index=task_def['classes'],
            columns=task_def['classes']
        )
        cm_df.to_csv(cm_dir / f"{task_def['name']}_confusion_matrix.csv", encoding='utf-8-sig')
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues')
        plt.title(f"{task_def['name_cn']} Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.tight_layout()
        plt.savefig(cm_dir / f"{task_def['name']}_confusion_matrix.png", dpi=150)
        plt.close()


def save_predictions(pred_labels, target_labels, valid_masks, image_names, output_dir):
    predictions_data = []
    
    for idx, img_name in enumerate(image_names):
        row = {'image_name': img_name}
        
        for i, (pred, target, mask) in enumerate(zip(pred_labels, target_labels, valid_masks)):
            task_def = config.TASK_DEFINITIONS[i]
            
            if mask[idx]:
                row[f"{task_def['name']}_pred"] = pred[idx]
                row[f"{task_def['name']}_true"] = target[idx]
                row[f"{task_def['name']}_pred_label"] = task_def['classes'][pred[idx]]
                row[f"{task_def['name']}_true_label"] = task_def['classes'][target[idx]]
            else:
                row[f"{task_def['name']}_pred"] = pred[idx]
                row[f"{task_def['name']}_true"] = -1
                row[f"{task_def['name']}_pred_label"] = task_def['classes'][pred[idx]]
                row[f"{task_def['name']}_true_label"] = "缺失"
        
        predictions_data.append(row)
    
    df = pd.DataFrame(predictions_data)
    df.to_csv(output_dir / "predictions.csv", index=False, encoding='utf-8-sig')
    
    return df


def create_visualization_samples(pred_labels, target_labels, valid_masks, image_names, output_dir, num_samples=20):
    vis_dir = output_dir / "visualizations"
    vis_dir.mkdir(exist_ok=True)
    
    correct_counts = []
    for idx in range(len(image_names)):
        correct = sum(
            1 for i, (pred, target, mask) in enumerate(zip(pred_labels, target_labels, valid_masks))
            if mask[idx] and pred[idx] == target[idx]
        )
        correct_counts.append((idx, correct))
    
    correct_counts.sort(key=lambda x: x[1])
    
    sample_indices = [idx for idx, _ in correct_counts[:num_samples]]
    
    for sample_idx, img_idx in enumerate(sample_indices):
        img_name = image_names[img_idx]
        img_path = config.IMAGE_DIR / img_name
        
        if not img_path.exists():
            continue
        
        img = Image.open(img_path).convert('RGB')
        
        info_text = f"Image: {img_name}\n\n"
        
        for i, (pred, target, mask) in enumerate(zip(pred_labels, target_labels, valid_masks)):
            task_def = config.TASK_DEFINITIONS[i]
            
            if mask[img_idx]:
                pred_label = task_def['classes'][pred[img_idx]]
                true_label = task_def['classes'][target[img_idx]]
                status = "✓" if pred[img_idx] == target[img_idx] else "✗"
                info_text += f"{task_def['name_cn']}: Pred={pred_label}, True={true_label} {status}\n"
            else:
                pred_label = task_def['classes'][pred[img_idx]]
                info_text += f"{task_def['name_cn']}: Pred={pred_label}, True=缺失\n"
        
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.imshow(img)
        ax.axis('off')
        ax.set_title(info_text, fontsize=10, loc='left', wrap=True)
        plt.tight_layout()
        plt.savefig(vis_dir / f"sample_{sample_idx:03d}.png", dpi=100, bbox_inches='tight')
        plt.close()


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    checkpoint_files = list(config.CHECKPOINT_DIR.glob("*.pth"))
    if not checkpoint_files:
        print("No checkpoint found! Please run train.py first.")
        return
    
    latest_checkpoint = max(checkpoint_files, key=os.path.getmtime)
    print(f"Loading checkpoint: {latest_checkpoint}")
    
    model = load_model(str(latest_checkpoint), device)
    
    data_loader = get_full_dataset_loader(
        csv_path=str(config.CSV_PATH),
        image_dir=str(config.IMAGE_DIR),
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS
    )
    
    print(f"Running inference on {len(data_loader.dataset)} samples...")
    pred_labels, target_labels, valid_masks, image_names = run_inference(model, data_loader, device)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = config.OUTPUT_DIR / f"inference_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    metrics = compute_all_metrics(pred_labels, target_labels, valid_masks)
    
    print("\n" + "=" * 60)
    print("Per-task Metrics:")
    print("=" * 60)
    
    metrics_summary = []
    for i, (task_def, task_metrics) in enumerate(zip(config.TASK_DEFINITIONS, metrics)):
        print(f"\n{task_def['name_cn']} ({task_def['name']}):")
        print(f"  Accuracy:  {task_metrics['accuracy']:.4f}")
        print(f"  F1 Score:  {task_metrics['f1']:.4f}")
        print(f"  Precision: {task_metrics['precision']:.4f}")
        print(f"  Recall:    {task_metrics['recall']:.4f}")
        print(f"  Valid Samples: {task_metrics['valid_samples']}")
        
        metrics_summary.append({
            'task_name': task_def['name'],
            'task_name_cn': task_def['name_cn'],
            'accuracy': task_metrics['accuracy'],
            'f1': task_metrics['f1'],
            'precision': task_metrics['precision'],
            'recall': task_metrics['recall'],
            'valid_samples': task_metrics['valid_samples']
        })
    
    avg_accuracy = np.mean([m['accuracy'] for m in metrics])
    avg_f1 = np.mean([m['f1'] for m in metrics])
    avg_precision = np.mean([m['precision'] for m in metrics])
    avg_recall = np.mean([m['recall'] for m in metrics])
    
    print("\n" + "=" * 60)
    print("Overall Metrics:")
    print("=" * 60)
    print(f"Average Accuracy:  {avg_accuracy:.4f}")
    print(f"Average F1 Score:  {avg_f1:.4f}")
    print(f"Average Precision: {avg_precision:.4f}")
    print(f"Average Recall:    {avg_recall:.4f}")
    
    overall_metrics = {
        'average_accuracy': avg_accuracy,
        'average_f1': avg_f1,
        'average_precision': avg_precision,
        'average_recall': avg_recall
    }
    
    metrics_df = pd.DataFrame(metrics_summary)
    metrics_df.to_csv(output_dir / "metrics_summary.csv", index=False, encoding='utf-8-sig')
    
    with open(output_dir / "overall_metrics.json", 'w', encoding='utf-8') as f:
        json.dump(overall_metrics, f, indent=2, ensure_ascii=False)
    
    print("\nSaving confusion matrices...")
    save_confusion_matrices(pred_labels, target_labels, valid_masks, output_dir)
    
    print("Saving predictions...")
    save_predictions(pred_labels, target_labels, valid_masks, image_names, output_dir)
    
    print("Creating visualization samples...")
    create_visualization_samples(pred_labels, target_labels, valid_masks, image_names, output_dir)
    
    print(f"\nInference completed!")
    print(f"Results saved to: {output_dir}")
    print(f"  - predictions.csv")
    print(f"  - metrics_summary.csv")
    print(f"  - overall_metrics.json")
    print(f"  - confusion_matrices/")
    print(f"  - visualizations/")


if __name__ == "__main__":
    main()
