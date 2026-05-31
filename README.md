# DNA-DetectLLM

一个基于DNA突变修复范式的零样本AI文本检测方法。

## 项目概述

DNA-DetectLLM 是一种创新的AI生成文本检测方法，灵感来源于DNA的突变修复机制。该方法无需训练即可检测多种大型语言模型生成的文本。

## 主要特性

-  **零样本检测**: 无需针对特定模型进行训练
-  **突变修复范式**: 基于DNA修复机制的原理
-  **跨域鲁棒性**: 在多种文本类型上表现稳定
-  **对抗鲁棒性**: 对常见攻击方法具有抵抗力
-  **多模型支持**: 支持GPT-4、Gemini、Claude等主流LLM

## 项目结构

```
DNA-DetectLLM/
├── Entropy-Gaze/          # 核心算法实现
│   ├── dna_detectllm/     # 检测器核心模块
│   ├── eval.py           # 评估脚本
│   └── main.py           # 主程序入口
├── Baselinse/            # 基线方法实现
├── Data/                 # 数据集
│   ├── Collected data/   # 收集的数据
│   ├── DetectRL/         # DetectRL基准数据
│   ├── M4/               # M4基准数据
│   ├── RealDet/          # RealDet基准数据
│   └── Text_attack/      # 对抗攻击样本
├── models/               # 模型文件（已忽略）
├── requirements.txt      # 依赖包列表
└── .gitignore           # Git忽略配置
```

## 快速开始

### 环境配置

```bash
# 安装依赖
pip install -r requirements.txt
```

### 基本使用

```python
# 导入检测器
from Entropy-Gaze.dna_detectllm.detector import DNADetectLLM

# 初始化检测器
detector = DNADetectLLM()

# 检测文本
text = "这是一段需要检测的文本..."
result = detector.detect(text)
print(f"AI生成概率: {result['ai_probability']:.4f}")
```

### 批量评估

```bash
# 评估在特定数据集上的性能
python Entropy-Gaze/eval.py --human_file=Data/Collected\ data/arxiv_human.json --ai_file=Data/Collected\ data/GPT4_machine_test.json
```

## 数据集

项目包含多个高质量检测基准数据集：

- **自收集数据**: 4,800篇人类写作文本，涵盖新闻、故事、学术写作
- **基准数据**: M4、DetectRL、RealDet等公开基准的采样数据
- **对抗样本**: 经过插入、删除、替换、改写等攻击处理的文本

## 实验结果

DNA-DetectLLM在多个基准测试中表现出色：
- 跨域检测准确率超过85%
- 对对抗攻击具有较强鲁棒性
- 在多种LLM生成的文本上表现一致

## 引用

如果您使用了本项目，请引用相关论文：

```bibtex
@article{dnadetectllm2024,
  title={DNA-DetectLLM: Zero-shot AI Text Detection via Mutation-Repair Paradigm},
  author={Your Name},
  journal={Conference/Journal Name},
  year={2024}
}
```
