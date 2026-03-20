# 舌象识别分类模型

基于ConvNeXtV2 Base的多任务舌象图像分类系统，可同时预测15个舌象属性。

## 项目结构

```
.
├── dataset.py          # 数据集定义和数据加载
├── model.py            # 模型定义（ConvNeXtV2 Base + 15个分类头）
├── train.py            # 训练脚本
├── evaluate.py         # 评估脚本
├── analyze.py          # 数据分析脚本
├── requirements.txt    # 依赖包
├── result.csv          # 训练数据标签
├── images/             # 图像目录
└── outputs/            # 训练输出目录
```

## 任务定义

15个舌象属性分类任务：

| 序号 | 任务名称 | 类别数 | 类别定义 |
|:---:|:---|:---:|:---|
| 1 | 舌体颜色 | 5 | [淡红, 白, 红, 绛, 紫] |
| 2 | 舌体特征 | 2 | [老, 嫩] |
| 3 | 舌体形状 | 3 | [胖大, 瘦小, 正常] |
| 4 | 齿痕 | 2 | [无齿痕, 有齿痕] |
| 5 | 点刺 | 2 | [无点刺, 有点刺] |
| 6 | 裂纹 | 2 | [无裂纹, 有裂纹] |
| 7 | 舌苔特征 | 3 | [舌苔厚, 舌苔薄, 舌苔正常] |
| 8 | 滑苔 | 2 | [无滑苔, 有滑苔] |
| 9 | 糙苔 | 2 | [无糙苔, 有糙苔] |
| 10 | 剥落苔 | 2 | [无剥落苔, 有剥落苔] |
| 11 | 腐苔 | 2 | [无腐苔, 有腐苔] |
| 12 | 腻苔 | 2 | [无腻苔, 有腻苔] |
| 13 | 舌苔颜色 | 4 | [白, 黄, 灰, 黑] |
| 14 | 舌尖红 | 2 | [无舌尖红, 有舌尖红] |
| 15 | 舌尖凹陷 | 2 | [无舌尖凹陷, 有舌尖凹陷] |

## 环境要求

- Python >= 3.8
- PyTorch >= 2.0.0
- timm >= 0.9.0
- 其他依赖见 requirements.txt

使用conda yolov11环境（已包含pytorch+timm+huggingface）：
```bash
conda activate yolov11
```

## 快速开始

### 1. 训练模型

```bash
python train.py \
    --csv_path result.csv \
    --image_dir images \
    --output_dir outputs \
    --epochs 50 \
    --batch_size 16 \
    --lr 1e-4 \
    --val_split 0.2
```

训练参数说明：
- `--csv_path`: 标签CSV文件路径
- `--image_dir`: 图像目录路径
- `--output_dir`: 输出目录路径
- `--epochs`: 训练轮数（默认50）
- `--batch_size`: 批次大小（默认16）
- `--lr`: 学习率（默认1e-4）
- `--val_split`: 验证集比例（默认0.2）
- `--dropout`: Dropout比率（默认0.3）

### 2. 评估模型

```bash
python evaluate.py \
    --checkpoint outputs/best_model.pth \
    --csv_path result.csv \
    --image_dir images \
    --output_dir evaluation_results \
    --visualize
```

评估输出：
- `predictions.csv`: 每个样本的预测结果
- `metrics.csv`: 每个任务的详细指标
- `overall_metrics.csv`: 总体指标
- `confusion_matrices/`: 每个任务的混淆矩阵（CSV和PNG）
- `visualizations/`: 预测结果可视化图像

### 3. 数据分析

```bash
python analyze.py \
    --csv_path result.csv \
    --image_dir images \
    --output_dir analysis_results \
    --checkpoint outputs/best_model.pth \
    --predictions_csv evaluation_results/predictions.csv
```

分析输出：
- `label_distribution/`: 每个任务的标签分布统计和可视化
- `tongue_color_samples/`: 舌体颜色样本可视化
- `feature_visualization/`: PCA和t-SNE特征可视化（需要checkpoint）
- `tongue_color_analysis/`: 舌体颜色混淆矩阵和错分样本分析（需要predictions_csv）

## 模型架构

- **Backbone**: ConvNeXtV2 Base (预训练)
- **分类头**: 15个独立的MLP分类头，每个任务一个
- **输入尺寸**: 224x224
- **数据增强**: Resize, RandomHorizontalFlip, RandomRotation, ColorJitter, Normalize

## 数据格式

CSV文件格式（第一列为图像文件名，后面15列为各任务标签）：
```csv
图片文件名,舌体颜色,舌体特征,舌体形状,齿痕,点刺,裂纹,舌苔特征,滑苔,糙苔,剥落苔,腐苔,腻苔,舌苔颜色,舌尖红,舌尖凹陷
image1.jpg,0,0,2,0,0,1,1,0,0,0,0,1,0,0,0
image2.jpg,2,1,2,0,1,0,1,0,0,0,0,1,0,0,0
...
```

- 标签为数值编码（0开始的整数）
- 缺失值用-1表示

## 评估指标

- 每个任务：Accuracy, Precision, Recall, F1-Score
- 总体：Average Accuracy, Average F1-Score

## 参考基线

- 平均准确率: ~0.7238
- 平均F1: ~0.6205

## 注意事项

1. 确保使用conda yolov11环境运行
2. CSV文件支持多种编码（utf-8, gbk, gb2312等）
3. 训练时自动处理缺失标签（-1）
4. 评估时跳过缺失标签的计算
