# 舌象识别分类模型

这是一个基于ConvNeXtV2的多任务舌象识别分类模型，支持同时预测15个舌象属性。

## 任务定义

模型同时预测以下15个舌象属性（标签数值0-5，缺失值为-1）：

1. 舌体颜色: [淡红, 白, 红, 绛, 紫]
2. 舌体特征: [老, 嫩]
3. 舌体形状: [胖大, 瘦小, 正常]
4. 齿痕: [无齿痕, 有齿痕]
5. 点刺: [无点刺, 有点刺]
6. 裂纹: [无裂纹, 有裂纹]
7. 舌苔特征: [舌苔厚, 舌苔薄, 舌苔正常]
8. 滑苔: [无滑苔, 有滑苔]
9. 糙苔: [无糙苔, 有糙苔]
10. 剥落苔: [无剥落苔, 有剥落苔]
11. 腐苔: [无腐苔, 有腐苔]
12. 腻苔: [无腻苔, 有腻苔]
13. 舌苔颜色: [白, 黄, 灰, 黑]
14. 舌尖红: [无舌尖红, 有舌尖红]
15. 舌尖凹陷: [无舌尖凹陷, 有舌尖凹陷]

## 环境要求

- Python 3.8+
- PyTorch >= 1.10.0
- CUDA (推荐)

主要依赖：
- timm (用于ConvNeXtV2模型)
- torchvision
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn

## 安装依赖

```bash
pip install -r requirements.txt
```

## 项目结构

```
t_ai/
├── config/
│   └── config.yaml          # 配置文件
├── src/
│   ├── __init__.py
│   ├── utils.py             # 工具函数
│   ├── model.py             # 模型定义
│   ├── dataset.py           # 数据集加载
│   ├── train.py             # 训练脚本
│   ├── infer.py             # 推理和评估
│   └── analyze.py           # 数据分析
├── images/                  # 图像数据目录
├── result.csv               # 训练标签CSV
├── checkpoints/             # 模型检查点
├── outputs/                 # 推理和评估输出
├── logs/                    # 训练日志
├── analysis/                # 分析结果
├── visualization/           # 可视化结果
├── main.py                  # 主入口
└── requirements.txt         # 依赖列表
```

## 使用方法

### 1. 数据准备

将舌象图片放在 `images/` 目录下，标签CSV文件命名为 `result.csv`，格式如下：

```
文件名,舌体颜色,舌体特征,舌体形状,齿痕,点刺,裂纹,舌苔特征,滑苔,糙苔,剥落苔,腐苔,腻苔,舌苔颜色,舌尖红,舌尖凹陷
image1.jpg,0,0,2,0,0,0,1,0,0,0,0,0,0,0,0
image2.jpg,2,1,0,1,0,1,2,0,0,0,0,1,1,0,0
...
```

### 2. 训练模型

```bash
python main.py train
```

训练过程中：
- 模型检查点保存在 `checkpoints/` 目录
- TensorBoard日志保存在 `logs/` 目录
- 训练日志保存在 `logs/train_*.log`

### 3. 评估模型

```bash
python main.py eval --checkpoint checkpoints/model_best.pth
```

评估输出（保存在 `outputs/` 目录）：
- `predictions.csv`: 所有样本的预测结果
- `evaluation_metrics.csv`: 各任务的准确率、F1、精确率、召回率
- `confusion_matrices/`: 各任务的混淆矩阵（CSV和图片）
- `visualization/`: 带预测结果的图片可视化

### 4. 推理新图像

```bash
python main.py infer --checkpoint checkpoints/model_best.pth --image_dir path/to/images
```

推理输出：
- `predictions.csv`: 预测结果
- `visualization/`: 可视化结果

### 5. 数据分析

```bash
# 基础分析（标签分布、样本可视化）
python main.py analyze

# 完整分析（含特征可视化和混淆分析，需要模型检查点）
python main.py analyze --checkpoint checkpoints/model_best.pth
```

分析内容：
1. 标签分布分析（每个任务的类别分布和缺失率）
2. 舌体颜色样本可视化
3. 特征提取后PCA + t-SNE可视化
4. 舌体颜色混淆矩阵和错分样本导出

## 配置说明

主要配置在 `config/config.yaml` 中：

- 数据配置：图像目录、CSV文件、批量大小、图像尺寸等
- 模型配置：ConvNeXtV2变体、 dropout率
- 训练配置：学习率、轮数、优化器参数
- 数据增强：训练和验证时的图像变换

## 模型结构

- 骨干网络：ConvNeXtV2 Base (预训练于ImageNet)
- 分类头：15个独立的线性层（LayerNorm + Linear）
- 损失函数：多任务交叉熵损失（自动忽略缺失标签）

## 参考指标

基线性能（供参考）：
- 平均准确率：~0.7238
- 平均F1：~0.6205

实际性能因数据和训练策略而异。

## 命令行参数

```
usage: main.py [-h] [--checkpoint CHECKPOINT] [--image_dir IMAGE_DIR]
               [--csv_file CSV_FILE] [--output_dir OUTPUT_DIR]
               [--config CONFIG]
               {train,eval,infer,analyze}

舌象识别分类模型

positional arguments:
  {train,eval,infer,analyze}
                        运行模式: train(训练), eval(评估), infer(推理), analyze(数据分析)

optional arguments:
  -h, --help            show this help message and exit
  --checkpoint CHECKPOINT
                        模型检查点路径(eval/infer/analyze模式需要)
  --image_dir IMAGE_DIR
                        推理时的图像目录
  --csv_file CSV_FILE   CSV标签文件路径
  --output_dir OUTPUT_DIR
                        输出目录
  --config CONFIG       配置文件路径
```

## License

MIT
