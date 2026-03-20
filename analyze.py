import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from datetime import datetime
import glob

import config
from dataset import get_full_dataset_loader
from model import load_model


def analyze_label_distribution(csv_path: str, output_dir):
    df = pd.read_csv(csv_path, encoding='gbk')
    
    label_cols = df.columns[1:1 + config.NUM_TASKS]
    
    dist_dir = output_dir / "label_distribution"
    dist_dir.mkdir(exist_ok=True)
    
    distribution_data = []
    
    for i, col in enumerate(label_cols):
        task_def = config.TASK_DEFINITIONS[i]
        
        valid_labels = df[col][df[col] != config.MISSING_LABEL]
        valid_labels = valid_labels.dropna()
        
        if len(valid_labels) == 0:
            continue
        
        label_counts = valid_labels.value_counts().sort_index()
        
        distribution_data.append({
            'task_name': task_def['name'],
            'task_name_cn': task_def['name_cn'],
            'total_samples': len(df),
            'valid_samples': len(valid_labels),
            'missing_samples': len(df) - len(valid_labels),
            'distribution': label_counts.to_dict()
        })
        
        plt.figure(figsize=(10, 6))
        x_labels = [task_def['classes'][idx] if idx < len(task_def['classes']) else f'Class_{idx}' 
                    for idx in label_counts.index]
        plt.bar(x_labels, label_counts.values)
        plt.title(f"{task_def['name_cn']} Distribution")
        plt.xlabel("Class")
        plt.ylabel("Count")
        plt.xticks(rotation=45, ha='right')
        
        for j, (idx, count) in enumerate(zip(label_counts.index, label_counts.values)):
            plt.text(j, count + 5, str(count), ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(dist_dir / f"{task_def['name']}_distribution.png", dpi=150)
        plt.close()
    
    summary_df = pd.DataFrame([{
        'Task': d['task_name_cn'],
        'Total': d['total_samples'],
        'Valid': d['valid_samples'],
        'Missing': d['missing_samples'],
        'Class Distribution': str(d['distribution'])
    } for d in distribution_data])
    summary_df.to_csv(dist_dir / "distribution_summary.csv", index=False, encoding='utf-8-sig')
    
    return distribution_data


def visualize_tongue_color_samples(csv_path: str, image_dir: str, output_dir, samples_per_class: int = 5):
    df = pd.read_csv(csv_path, encoding='gbk')
    
    color_col = df.columns[1]
    task_def = config.TASK_DEFINITIONS[0]
    
    vis_dir = output_dir / "tongue_color_samples"
    vis_dir.mkdir(exist_ok=True)
    
    for class_idx, class_name in enumerate(task_def['classes']):
        class_samples = df[df[color_col] == class_idx]
        
        if len(class_samples) == 0:
            continue
        
        sample_images = class_samples.sample(min(samples_per_class, len(class_samples)))
        
        fig, axes = plt.subplots(1, len(sample_images), figsize=(15, 3))
        if len(sample_images) == 1:
            axes = [axes]
        
        for ax, (_, row) in zip(axes, sample_images.iterrows()):
            img_name = row.iloc[0]
            img_path = os.path.join(image_dir, img_name)
            
            if os.path.exists(img_path):
                img = Image.open(img_path)
                ax.imshow(img)
            ax.axis('off')
            ax.set_title(img_name[:15] + '...' if len(img_name) > 15 else img_name, fontsize=8)
        
        plt.suptitle(f"{class_name} (Class {class_idx})", fontsize=12)
        plt.tight_layout()
        plt.savefig(vis_dir / f"class_{class_idx}_{class_name}.png", dpi=150)
        plt.close()


def extract_features(model, data_loader, device):
    model.eval()
    
    all_features = []
    all_labels = [[] for _ in range(config.NUM_TASKS)]
    all_image_names = []
    
    with torch.no_grad():
        pbar = tqdm(data_loader, desc="Extracting features")
        for images, labels, img_names in pbar:
            images = images.to(device)
            
            features = model.get_features(images)
            
            all_features.append(features.cpu().numpy())
            
            for i, label in enumerate(labels):
                all_labels[i].append(label.numpy())
            
            all_image_names.extend(img_names)
    
    all_features = np.concatenate(all_features, axis=0)
    all_labels = [np.concatenate(l, axis=0) for l in all_labels]
    
    return all_features, all_labels, all_image_names


def visualize_pca_tsne(features, labels, output_dir):
    vis_dir = output_dir / "feature_visualization"
    vis_dir.mkdir(exist_ok=True)
    
    print("Running PCA...")
    pca = PCA(n_components=2)
    features_pca = pca.fit_transform(features)
    
    print("Running t-SNE...")
    tsne = TSNE(n_components=2, random_state=config.SEED, perplexity=min(30, len(features) - 1))
    features_tsne = tsne.fit_transform(features)
    
    target_labels = [np.argmax(l, axis=1) for l in labels]
    valid_masks = [l.sum(axis=1) > 0 for l in labels]
    
    for i, (task_def, target, mask) in enumerate(zip(config.TASK_DEFINITIONS, target_labels, valid_masks)):
        if mask.sum() < 2:
            continue
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        for ax, features_2d, title in zip(axes, [features_pca, features_tsne], ['PCA', 't-SNE']):
            valid_features = features_2d[mask]
            valid_labels = target[mask]
            
            scatter = ax.scatter(
                valid_features[:, 0],
                valid_features[:, 1],
                c=valid_labels,
                cmap='tab10',
                alpha=0.6,
                s=10
            )
            ax.set_title(f"{task_def['name_cn']} - {title}")
            ax.set_xlabel(f"{title} 1")
            ax.set_ylabel(f"{title} 2")
            
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_ticks(range(task_def['num_classes']))
            cbar.set_ticklabels(task_def['classes'])
        
        plt.tight_layout()
        plt.savefig(vis_dir / f"{task_def['name']}_pca_tsne.png", dpi=150)
        plt.close()


def export_tongue_color_errors(pred_labels, target_labels, valid_masks, image_names, output_dir):
    task_def = config.TASK_DEFINITIONS[0]
    pred = pred_labels[0]
    target = target_labels[0]
    mask = valid_masks[0]
    
    error_indices = np.where((mask) & (pred != target))[0]
    
    if len(error_indices) == 0:
        print("No errors found for tongue color task.")
        return
    
    error_dir = output_dir / "tongue_color_errors"
    error_dir.mkdir(exist_ok=True)
    
    error_data = []
    for idx in error_indices:
        error_data.append({
            'image_name': image_names[idx],
            'true_class': task_def['classes'][target[idx]],
            'true_idx': target[idx],
            'pred_class': task_def['classes'][pred[idx]],
            'pred_idx': pred[idx]
        })
    
    error_df = pd.DataFrame(error_data)
    error_df.to_csv(error_dir / "error_samples.csv", index=False, encoding='utf-8-sig')
    
    cm_dir = output_dir / "confusion_matrices"
    cm_file = cm_dir / f"{task_def['name']}_confusion_matrix.csv"
    
    if cm_file.exists():
        cm = pd.read_csv(cm_file, index_col=0)
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Reds')
        plt.title(f"{task_def['name_cn']} Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.tight_layout()
        plt.savefig(error_dir / "tongue_color_confusion_matrix.png", dpi=150)
        plt.close()
    
    num_vis = min(20, len(error_indices))
    fig, axes = plt.subplots(4, 5, figsize=(15, 12))
    axes = axes.flatten()
    
    for i, idx in enumerate(error_indices[:num_vis]):
        img_name = image_names[idx]
        img_path = config.IMAGE_DIR / img_name
        
        if os.path.exists(img_path):
            img = Image.open(img_path)
            axes[i].imshow(img)
        axes[i].axis('off')
        axes[i].set_title(
            f"T:{task_def['classes'][target[idx]]}\nP:{task_def['classes'][pred[idx]]}",
            fontsize=8
        )
    
    for i in range(num_vis, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle("Tongue Color Misclassified Samples (T=True, P=Pred)", fontsize=12)
    plt.tight_layout()
    plt.savefig(error_dir / "error_samples_visualization.png", dpi=150)
    plt.close()
    
    print(f"Exported {len(error_indices)} error samples to {error_dir}")


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = config.OUTPUT_DIR / f"analysis_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 60)
    print("1. Analyzing label distribution...")
    print("=" * 60)
    distribution_data = analyze_label_distribution(str(config.CSV_PATH), output_dir)
    
    print("\n" + "=" * 60)
    print("2. Visualizing tongue color samples...")
    print("=" * 60)
    visualize_tongue_color_samples(str(config.CSV_PATH), str(config.IMAGE_DIR), output_dir)
    
    checkpoint_files = list(config.CHECKPOINT_DIR.glob("*.pth"))
    if checkpoint_files:
        latest_checkpoint = max(checkpoint_files, key=os.path.getmtime)
        print(f"\nLoading checkpoint: {latest_checkpoint}")
        
        model = load_model(str(latest_checkpoint), device)
        
        data_loader = get_full_dataset_loader(
            csv_path=str(config.CSV_PATH),
            image_dir=str(config.IMAGE_DIR),
            batch_size=config.BATCH_SIZE,
            num_workers=config.NUM_WORKERS
        )
        
        print("\n" + "=" * 60)
        print("3. Extracting features and visualizing PCA/t-SNE...")
        print("=" * 60)
        features, labels, image_names = extract_features(model, data_loader, device)
        visualize_pca_tsne(features, labels, output_dir)
        
        print("\n" + "=" * 60)
        print("4. Running inference for error analysis...")
        print("=" * 60)
        
        model.eval()
        all_predictions = [[] for _ in range(config.NUM_TASKS)]
        all_targets = [[] for _ in range(config.NUM_TASKS)]
        
        with torch.no_grad():
            for images, batch_labels, _ in tqdm(data_loader, desc="Inference"):
                images = images.to(device)
                outputs = model(images)
                
                for i, output in enumerate(outputs):
                    all_predictions[i].append(output.cpu().numpy())
                
                for i, label in enumerate(batch_labels):
                    all_targets[i].append(label.numpy())
        
        all_predictions = [np.concatenate(p, axis=0) for p in all_predictions]
        all_targets = [np.concatenate(t, axis=0) for t in all_targets]
        
        pred_labels = [np.argmax(p, axis=1) for p in all_predictions]
        target_labels = [np.argmax(t, axis=1) for t in all_targets]
        valid_masks = [t.sum(axis=1) > 0 for t in all_targets]
        
        print("\n" + "=" * 60)
        print("5. Exporting tongue color errors...")
        print("=" * 60)
        export_tongue_color_errors(pred_labels, target_labels, valid_masks, image_names, output_dir)
    else:
        print("\nNo checkpoint found. Skipping feature extraction and error analysis.")
        print("Run train.py first to generate a checkpoint.")
    
    print(f"\nAnalysis completed!")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
