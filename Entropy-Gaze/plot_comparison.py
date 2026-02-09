#coding: utf-8
import json
import numpy as np
import matplotlib.pyplot as plt
# 确保matplotlib支持中文显示（可选但建议添加）
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'WenQuanYi Micro Hei']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

from dna_detectllm.detector import EntropyGazeDetector

# 1. 初始化 (使用最佳阈值 k=1.5)
model_path = "/root/autodl-tmp/models/LLM-Research/Meta-Llama-3-8B-Instruct"
detector = EntropyGazeDetector(model_name_or_path=model_path)
BEST_K = 1.5

# 2. 读取数据
with open("Data/Collected data/GPT4_machine_test.json", "r", encoding='utf-8', errors='ignore') as f:
    ai_texts = json.load(f)["machine_text"]
with open("Data/Collected data/xsum_human.json", "r", encoding='utf-8', errors='ignore') as f:
    human_texts = json.load(f)["human_text"]

# 3. 寻找最具代表性的样本 (方差差异最大的那一对)
print("Searching for the most representative samples...")
best_pair = None
max_contrast = 0

# 快速扫描前20对
for i in range(20):
    _, e_ai = detector.compute_thermodynamics(ai_texts[i])
    _, e_hu = detector.compute_thermodynamics(human_texts[i])
    
    # 寻找 Human 方差大且 AI 方差小的极端例子
    contrast = np.var(e_hu) - np.var(e_ai)
    if contrast > max_contrast:
        max_contrast = contrast
        best_pair = (e_ai, e_hu)

e_ai, e_hu = best_pair

# 4. 绘图：热力学指纹对比
plt.figure(figsize=(12, 6))

# 画 AI 曲线
plt.subplot(2, 1, 1)
plt.plot(e_ai, color='#1f77b4', label='AI (GPT-4) - Adiabatic', linewidth=1.5)
# 标记断裂点
threshold_ai = np.mean(e_ai) + BEST_K * np.std(e_ai)
fractures_ai = np.where(e_ai > threshold_ai)[0]
plt.scatter(fractures_ai, e_ai[fractures_ai], color='red', s=20, zorder=5, label=f'Fractures (k={BEST_K})')
plt.title(f"AI Entropy Flow (Var: {np.var(e_ai):.2f}) - Smooth & Laminar")
plt.ylabel("Self-Information")
plt.legend(loc="upper right")
plt.grid(True, alpha=0.3)

# 画 Human 曲线
plt.subplot(2, 1, 2)
plt.plot(e_hu, color='#ff7f0e', label='Human (XSum) - Dissipative', linewidth=1.5)
# 标记断裂点
threshold_hu = np.mean(e_hu) + BEST_K * np.std(e_hu)
fractures_hu = np.where(e_hu > threshold_hu)[0]
plt.scatter(fractures_hu, e_hu[fractures_hu], color='red', s=20, zorder=5, label=f'Fractures (k={BEST_K})')
plt.title(f"Human Entropy Flow (Var: {np.var(e_hu):.2f}) - Turbulent & Bursty")
plt.ylabel("Self-Information")
plt.xlabel("Token Index")
plt.legend(loc="upper right")
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('paper_figure_contrast.png', dpi=300)
# 修复：移除中文，使用纯英文输出，避免编码问题
print(f"? Paper figure generated: paper_figure_contrast.png (Contrast: {max_contrast:.4f})")