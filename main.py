# -*- coding: utf-8 -*-
"""
舌象识别分类模型主入口
支持训练、评估、推理、分析四种模式
"""
import os
import sys
import argparse

# 添加src路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    parser = argparse.ArgumentParser(description="舌象识别分类模型")
    parser.add_argument(
        "mode", 
        type=str, 
        choices=["train", "eval", "infer", "analyze"],
        help="运行模式: train(训练), eval(评估), infer(推理), analyze(数据分析)"
    )
    parser.add_argument("--checkpoint", type=str, help="模型检查点路径(eval/infer/analyze模式需要)")
    parser.add_argument("--image_dir", type=str, help="推理时的图像目录")
    parser.add_argument("--csv_file", type=str, help="CSV标签文件路径")
    parser.add_argument("--output_dir", type=str, help="输出目录")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="配置文件路径")
    
    args = parser.parse_args()
    
    if args.mode == "train":
        from src.train import main as train_main
        print("=" * 80)
        print("开始训练舌象识别分类模型")
        print("=" * 80)
        train_main()
    
    elif args.mode == "eval":
        if not args.checkpoint:
            print("错误: eval模式需要提供--checkpoint参数")
            sys.exit(1)
        
        from src.infer import main_evaluate
        print("=" * 80)
        print("开始评估舌象识别分类模型")
        print("=" * 80)
        main_evaluate(args.checkpoint, args.csv_file, args.output_dir)
    
    elif args.mode == "infer":
        if not args.checkpoint:
            print("错误: infer模式需要提供--checkpoint参数")
            sys.exit(1)
        
        from src.infer import main_inference
        print("=" * 80)
        print("开始推理舌象识别")
        print("=" * 80)
        main_inference(args.checkpoint, args.image_dir, None, args.output_dir)
    
    elif args.mode == "analyze":
        from src.analyze import main_analysis
        print("=" * 80)
        print("开始数据分析")
        print("=" * 80)
        main_analysis(args.checkpoint, args.csv_file, args.output_dir)


if __name__ == "__main__":
    main()
