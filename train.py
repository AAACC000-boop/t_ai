import os
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import json
from datetime import datetime

import config
from dataset import get_dataloaders
from model import MultiHeadTongueModel, MultiTaskLoss, save_model


def compute_metrics(predictions: list, targets: list, num_classes_list: list):
    all_metrics = []
    
    for i, (pred, target, num_classes) in enumerate(zip(predictions, targets, num_classes_list)):
        valid_mask = target.sum(axis=1) > 0
        
        if valid_mask.sum() == 0:
            all_metrics.append({
                'accuracy': 0.0,
                'f1': 0.0,
                'precision': 0.0,
                'recall': 0.0
            })
            continue
        
        pred_valid = pred[valid_mask]
        target_valid = target[valid_mask]
        
        pred_labels = np.argmax(pred_valid, axis=1)
        target_labels = np.argmax(target_valid, axis=1)
        
        metrics = {
            'accuracy': accuracy_score(target_labels, pred_labels),
            'f1': f1_score(target_labels, pred_labels, average='weighted', zero_division=0),
            'precision': precision_score(target_labels, pred_labels, average='weighted', zero_division=0),
            'recall': recall_score(target_labels, pred_labels, average='weighted', zero_division=0)
        }
        all_metrics.append(metrics)
    
    return all_metrics


def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    all_predictions = [[] for _ in range(config.NUM_TASKS)]
    all_targets = [[] for _ in range(config.NUM_TASKS)]
    
    pbar = tqdm(train_loader, desc="Training")
    for images, labels, _ in pbar:
        images = images.to(device)
        labels = [label.to(device) for label in labels]
        
        optimizer.zero_grad()
        
        outputs = model(images)
        
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        for i, (output, label) in enumerate(zip(outputs, labels)):
            all_predictions[i].append(output.detach().cpu().numpy())
            all_targets[i].append(label.detach().cpu().numpy())
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    avg_loss = total_loss / len(train_loader)
    
    all_predictions = [np.concatenate(p, axis=0) for p in all_predictions]
    all_targets = [np.concatenate(t, axis=0) for t in all_targets]
    
    metrics = compute_metrics(all_predictions, all_targets, config.NUM_CLASSES_PER_TASK)
    
    return avg_loss, metrics


def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_predictions = [[] for _ in range(config.NUM_TASKS)]
    all_targets = [[] for _ in range(config.NUM_TASKS)]
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Validating")
        for images, labels, _ in pbar:
            images = images.to(device)
            labels = [label.to(device) for label in labels]
            
            outputs = model(images)
            
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            for i, (output, label) in enumerate(zip(outputs, labels)):
                all_predictions[i].append(output.cpu().numpy())
                all_targets[i].append(label.cpu().numpy())
    
    avg_loss = total_loss / len(val_loader)
    
    all_predictions = [np.concatenate(p, axis=0) for p in all_predictions]
    all_targets = [np.concatenate(t, axis=0) for t in all_targets]
    
    metrics = compute_metrics(all_predictions, all_targets, config.NUM_CLASSES_PER_TASK)
    
    return avg_loss, metrics


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    train_loader, val_loader = get_dataloaders(
        csv_path=str(config.CSV_PATH),
        image_dir=str(config.IMAGE_DIR),
        batch_size=config.BATCH_SIZE,
        val_split=config.VAL_SPLIT,
        num_workers=config.NUM_WORKERS,
        seed=config.SEED
    )
    
    model = MultiHeadTongueModel(
        model_name="convnextv2_base",
        pretrained=True,
        num_classes_per_task=config.NUM_CLASSES_PER_TASK
    )
    model = model.to(device)
    
    criterion = MultiTaskLoss()
    
    optimizer = AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )
    
    scheduler = CosineAnnealingLR(optimizer, T_max=config.NUM_EPOCHS, eta_min=1e-6)
    
    best_f1 = 0.0
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_metrics': [],
        'val_metrics': []
    }
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_path = config.CHECKPOINT_DIR / f"best_model_{timestamp}.pth"
    
    print(f"\nStarting training for {config.NUM_EPOCHS} epochs...")
    
    for epoch in range(config.NUM_EPOCHS):
        print(f"\nEpoch {epoch + 1}/{config.NUM_EPOCHS}")
        print("-" * 50)
        
        train_loss, train_metrics = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_metrics = validate(model, val_loader, criterion, device)
        
        scheduler.step()
        
        avg_train_f1 = np.mean([m['f1'] for m in train_metrics])
        avg_val_f1 = np.mean([m['f1'] for m in val_metrics])
        avg_train_acc = np.mean([m['accuracy'] for m in train_metrics])
        avg_val_acc = np.mean([m['accuracy'] for m in val_metrics])
        
        print(f"Train Loss: {train_loss:.4f}, Avg Acc: {avg_train_acc:.4f}, Avg F1: {avg_train_f1:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Avg Acc: {avg_val_acc:.4f}, Avg F1: {avg_val_f1:.4f}")
        
        print("\nPer-task validation metrics:")
        for i, (task_def, metrics) in enumerate(zip(config.TASK_DEFINITIONS, val_metrics)):
            print(f"  {task_def['name_cn']}: Acc={metrics['accuracy']:.4f}, F1={metrics['f1']:.4f}")
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_metrics'].append(train_metrics)
        history['val_metrics'].append(val_metrics)
        
        if avg_val_f1 > best_f1:
            best_f1 = avg_val_f1
            save_model(model, optimizer, epoch, best_f1, str(checkpoint_path))
            print(f"New best F1: {best_f1:.4f}")
    
    history_path = config.OUTPUT_DIR / f"training_history_{timestamp}.json"
    with open(history_path, 'w', encoding='utf-8') as f:
        serializable_history = {
            'train_loss': history['train_loss'],
            'val_loss': history['val_loss'],
            'train_metrics': [[{k: float(v) for k, v in m.items()} for m in metrics] for metrics in history['train_metrics']],
            'val_metrics': [[{k: float(v) for k, v in m.items()} for m in metrics] for metrics in history['val_metrics']]
        }
        json.dump(serializable_history, f, ensure_ascii=False, indent=2)
    
    print(f"\nTraining completed!")
    print(f"Best validation F1: {best_f1:.4f}")
    print(f"Best model saved to: {checkpoint_path}")
    print(f"Training history saved to: {history_path}")


if __name__ == "__main__":
    main()
