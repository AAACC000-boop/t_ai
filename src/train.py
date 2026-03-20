# -*- coding: utf-8 -*-
"""
训练脚本
"""
import os
import sys
import time
import logging
from typing import Dict, List, Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np

# 添加src路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import (
    load_config, set_seed, get_device, create_dirs,
    save_checkpoint, load_checkpoint, AverageMeter, MetricMeter
)
from src.model import create_model, create_loss_fn
from src.dataset import create_dataloaders


def setup_logging(config: Dict[str, Any]) -> logging.Logger:
    """
    设置日志
    """
    log_dir = config["paths"]["log_dir"]
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    log_file = os.path.join(log_dir, f"train_{timestamp}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)


def create_optimizer(config: Dict[str, Any], model: nn.Module) -> optim.Optimizer:
    """
    创建优化器
    """
    opt_config = config["optimizer"]
    train_config = config["training"]
    
    optimizer = getattr(optim, opt_config["name"])(
        model.parameters(),
        lr=train_config["learning_rate"],
        weight_decay=train_config["weight_decay"],
        betas=tuple(opt_config["betas"]),
        eps=opt_config["eps"]
    )
    
    return optimizer


def create_scheduler(config: Dict[str, Any], optimizer: optim.Optimizer) -> Any:
    """
    创建学习率调度器
    """
    sched_config = config["scheduler"]
    
    scheduler = getattr(optim.lr_scheduler, sched_config["name"])(
        optimizer,
        T_max=sched_config["T_max"],
        eta_min=sched_config["eta_min"]
    )
    
    return scheduler


def calculate_accuracy(
    logits_list: List[torch.Tensor],
    targets_list: List[torch.Tensor]
) -> List[float]:
    """
    计算每个任务的准确率
    """
    accuracies = []
    
    for logits, targets in zip(logits_list, targets_list):
        # 获取有效样本掩码
        if targets.dim() > 1:
            # one-hot格式
            valid_mask = targets.sum(dim=1) > 0
            targets = targets.argmax(dim=1)
        else:
            valid_mask = targets != -1
        
        num_valid = valid_mask.sum().item()
        if num_valid == 0:
            accuracies.append(0.0)
            continue
        
        # 计算预测
        predictions = logits.argmax(dim=1)
        
        # 只计算有效样本
        valid_predictions = predictions[valid_mask]
        valid_targets = targets[valid_mask]
        
        correct = (valid_predictions == valid_targets).sum().item()
        accuracy = correct / num_valid
        accuracies.append(accuracy)
    
    return accuracies


def train_one_epoch(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    loss_fn: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
    config: Dict[str, Any],
    logger: logging.Logger,
    writer: SummaryWriter
) -> Dict[str, Any]:
    """
    训练一个epoch
    """
    model.train()
    
    num_tasks = config["task"]["num_tasks"]
    loss_meter = AverageMeter()
    acc_meter = MetricMeter(num_tasks)
    task_loss_meters = [AverageMeter() for _ in range(num_tasks)]
    
    pbar = tqdm(train_loader, desc=f"训练 Epoch {epoch + 1}")
    
    for batch_idx, (images, targets_list, _) in enumerate(pbar):
        images = images.to(device)
        targets_list = [t.to(device) for t in targets_list]
        
        # 前向传播
        logits_list = model(images)
        
        # 计算损失
        losses = loss_fn(logits_list, targets_list)
        total_loss = losses["total"]
        
        # 反向传播
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        # 更新指标
        batch_size = images.size(0)
        loss_meter.update(total_loss.item(), batch_size)
        
        # 更新各任务损失
        for i in range(num_tasks):
            task_loss_meters[i].update(losses[f"task_{i}"].item(), batch_size)
        
        # 计算准确率
        accuracies = calculate_accuracy(logits_list, targets_list)
        acc_meter.update(accuracies, batch_size)
        
        # 更新进度条
        pbar.set_postfix({
            "损失": f"{loss_meter.avg:.4f}",
            "平均准确率": f"{np.mean(acc_meter.get_averages()):.4f}"
        })
        
        # 日志记录
        if (batch_idx + 1) % config["training"]["log_freq"] == 0:
            global_step = epoch * len(train_loader) + batch_idx
            writer.add_scalar("train/loss", loss_meter.avg, global_step)
            for i in range(num_tasks):
                writer.add_scalar(f"train/task_{i}_loss", task_loss_meters[i].avg, global_step)
                writer.add_scalar(f"train/task_{i}_acc", acc_meter.meters[i].avg, global_step)
    
    # 计算平均指标
    avg_loss = loss_meter.avg
    avg_accs = acc_meter.get_averages()
    mean_acc = np.mean(avg_accs)
    
    # 记录日志
    logger.info(f"训练 Epoch {epoch + 1}: 平均损失={avg_loss:.4f}, 平均准确率={mean_acc:.4f}")
    for i, acc in enumerate(avg_accs):
        logger.info(f"  任务{i} ({config['task']['task_names'][i]}): {acc:.4f}")
    
    return {
        "loss": avg_loss,
        "accuracies": avg_accs,
        "mean_accuracy": mean_acc
    }


def validate(
    model: nn.Module,
    val_loader: torch.utils.data.DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    epoch: int,
    config: Dict[str, Any],
    logger: logging.Logger,
    writer: SummaryWriter
) -> Dict[str, Any]:
    """
    验证
    """
    model.eval()
    
    num_tasks = config["task"]["num_tasks"]
    loss_meter = AverageMeter()
    acc_meter = MetricMeter(num_tasks)
    task_loss_meters = [AverageMeter() for _ in range(num_tasks)]
    
    all_logits = [[] for _ in range(num_tasks)]
    all_targets = [[] for _ in range(num_tasks)]
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc=f"验证 Epoch {epoch + 1}")
        
        for images, targets_list, _ in pbar:
            images = images.to(device)
            targets_list = [t.to(device) for t in targets_list]
            
            # 前向传播
            logits_list = model(images)
            
            # 计算损失
            losses = loss_fn(logits_list, targets_list)
            total_loss = losses["total"]
            
            # 更新指标
            batch_size = images.size(0)
            loss_meter.update(total_loss.item(), batch_size)
            
            # 更新各任务损失
            for i in range(num_tasks):
                task_loss_meters[i].update(losses[f"task_{i}"].item(), batch_size)
            
            # 计算准确率
            accuracies = calculate_accuracy(logits_list, targets_list)
            acc_meter.update(accuracies, batch_size)
            
            # 收集预测结果用于计算F1等指标
            for i in range(num_tasks):
                # 获取有效样本
                targets = targets_list[i]
                if targets.dim() > 1:
                    valid_mask = targets.sum(dim=1) > 0
                    targets = targets.argmax(dim=1)
                else:
                    valid_mask = targets != -1
                
                if valid_mask.sum() > 0:
                    predictions = logits_list[i].argmax(dim=1)
                    all_logits[i].append(logits_list[i][valid_mask].cpu())
                    all_targets[i].append(targets[valid_mask].cpu())
            
            # 更新进度条
            pbar.set_postfix({
                "损失": f"{loss_meter.avg:.4f}",
                "平均准确率": f"{np.mean(acc_meter.get_averages()):.4f}"
            })
    
    # 计算平均指标
    avg_loss = loss_meter.avg
    avg_accs = acc_meter.get_averages()
    mean_acc = np.mean(avg_accs)
    
    # 记录到TensorBoard
    writer.add_scalar("val/loss", avg_loss, epoch)
    for i in range(num_tasks):
        writer.add_scalar(f"val/task_{i}_loss", task_loss_meters[i].avg, epoch)
        writer.add_scalar(f"val/task_{i}_acc", avg_accs[i], epoch)
    
    # 记录日志
    logger.info(f"验证 Epoch {epoch + 1}: 平均损失={avg_loss:.4f}, 平均准确率={mean_acc:.4f}")
    for i, acc in enumerate(avg_accs):
        logger.info(f"  任务{i} ({config['task']['task_names'][i]}): {acc:.4f}")
    
    return {
        "loss": avg_loss,
        "accuracies": avg_accs,
        "mean_accuracy": mean_acc,
        "all_logits": all_logits,
        "all_targets": all_targets
    }


def main():
    """
    主训练函数
    """
    # 加载配置
    config = load_config()
    
    # 创建目录
    create_dirs(config)
    
    # 设置日志
    logger = setup_logging(config)
    
    # 设置随机种子
    set_seed(config["data"]["random_seed"])
    
    # 获取设备
    device = get_device()
    logger.info(f"使用设备: {device}")
    
    # 创建TensorBoard writer
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    log_dir = os.path.join(config["paths"]["log_dir"], f"runs_{timestamp}")
    writer = SummaryWriter(log_dir=log_dir)
    
    # 创建数据加载器
    logger.info("创建数据加载器...")
    train_loader, val_loader = create_dataloaders(config)
    logger.info(f"训练集大小: {len(train_loader.dataset)}, 验证集大小: {len(val_loader.dataset)}")
    
    # 创建模型
    logger.info("创建模型...")
    model = create_model(config)
    model = model.to(device)
    
    # 创建损失函数
    loss_fn = create_loss_fn(config)
    loss_fn = loss_fn.to(device)
    
    # 创建优化器和调度器
    optimizer = create_optimizer(config, model)
    scheduler = create_scheduler(config, optimizer)
    
    # 训练配置
    train_config = config["training"]
    num_epochs = train_config["epochs"]
    patience = train_config["early_stopping_patience"]
    
    # 初始化训练状态
    best_mean_acc = 0.0
    best_epoch = 0
    early_stop_count = 0
    start_epoch = 0
    
    # 检查是否需要恢复训练
    checkpoint_path = os.path.join(config["paths"]["checkpoint_dir"], "checkpoint_latest.pth")
    if os.path.exists(checkpoint_path):
        logger.info(f"加载检查点: {checkpoint_path}")
        checkpoint = load_checkpoint(checkpoint_path, model, optimizer, scheduler, device)
        start_epoch = checkpoint["epoch"] + 1
        best_mean_acc = checkpoint["metrics"].get("mean_accuracy", 0.0)
        logger.info(f"从第 {start_epoch} 轮恢复训练，最佳准确率: {best_mean_acc:.4f}")
    
    # 训练循环
    logger.info(f"开始训练，共 {num_epochs} 轮...")
    
    for epoch in range(start_epoch, num_epochs):
        # 训练
        train_metrics = train_one_epoch(
            model, train_loader, loss_fn, optimizer, device, epoch, config, logger, writer
        )
        
        # 验证
        val_metrics = validate(
            model, val_loader, loss_fn, device, epoch, config, logger, writer
        )
        
        # 更新学习率
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        writer.add_scalar("train/lr", current_lr, epoch)
        logger.info(f"学习率: {current_lr:.8f}")
        
        # 保存最新检查点
        latest_path = os.path.join(config["paths"]["checkpoint_dir"], "checkpoint_latest.pth")
        save_checkpoint(
            model, optimizer, scheduler, epoch,
            {"train": train_metrics, "val": val_metrics, "mean_accuracy": val_metrics["mean_accuracy"]},
            latest_path
        )
        
        # 保存定期检查点
        if (epoch + 1) % train_config["save_freq"] == 0:
            save_path = os.path.join(config["paths"]["checkpoint_dir"], f"checkpoint_epoch_{epoch + 1}.pth")
            save_checkpoint(
                model, optimizer, scheduler, epoch,
                {"train": train_metrics, "val": val_metrics, "mean_accuracy": val_metrics["mean_accuracy"]},
                save_path
            )
            logger.info(f"检查点已保存: {save_path}")
        
        # 保存最佳模型
        current_mean_acc = val_metrics["mean_accuracy"]
        if current_mean_acc > best_mean_acc:
            best_mean_acc = current_mean_acc
            best_epoch = epoch + 1
            early_stop_count = 0
            
            if train_config["save_best"]:
                best_path = os.path.join(config["paths"]["checkpoint_dir"], "checkpoint_latest.pth")
                save_checkpoint(
                    model, optimizer, scheduler, epoch,
                    {"train": train_metrics, "val": val_metrics, "mean_accuracy": val_metrics["mean_accuracy"]},
                    best_path, is_best=True
                )
                logger.info(f"新最佳模型已保存! 平均准确率: {best_mean_acc:.4f}")
        else:
            early_stop_count += 1
            logger.info(f"无改善，连续 {early_stop_count}/{patience} 轮")
            
            # 早停
            if early_stop_count >= patience:
                logger.info(f"早停触发! 最佳轮次: {best_epoch}, 最佳平均准确率: {best_mean_acc:.4f}")
                break
        
        logger.info("-" * 80)
    
    logger.info(f"训练完成! 最佳轮次: {best_epoch}, 最佳平均准确率: {best_mean_acc:.4f}")
    writer.close()


if __name__ == "__main__":
    main()
