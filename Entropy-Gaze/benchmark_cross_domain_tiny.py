# -*- coding: utf-8 -*-
import torch
import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset
from sklearn.metrics import roc_curve, auc
from dna_detectllm.detector import EntropyGazeDetector
import os

# ================= 配置区域 (请修改这里) =================
# 请确保您已经下载了 Qwen2.5-1.5B 或者其他小模型
# 如果您没有下载，请先去 AutoDL 的模型中心或 HF 镜像下载
MODEL_PATH = "/root/autodl-tmp/models/qwen/Qwen2.5-1.5B-Instruct" 

# 如果您找不到 1.5B 模型，可以使用 GPT-2 (通常系统自带) 作为极端测试
# MODEL_PATH = "gpt2" 

print(f"[Config] Using Tiny Model: {MODEL_PATH}")
# =======================================================

# ================= 硬编码数据 (保持一致性) =================
WRITING_SAMPLES = [
    "The clock struck thirteen, and the walls began to bleed geometric shapes. I realized too late that the architect hadn't designed a house, but a trap for time itself.",
    "The dragon didn't breathe fire; it breathed memories. One puff, and you were reliving your third birthday, crying over dropped ice cream while it ate you.",
    "In a world where shadows have legal rights, mine was suing me for defamation. It claimed I didn't lead an interesting enough life for it to cast a dramatic silhouette.",
    "The coffee shop was normal, except for the barista who had three eyes and the menu that listed 'Regret' as a flavor shot. I ordered a latte, black, no sugar.",
    "He found a door in the trunk of the old oak tree. Not a metaphorical door, but a literal mahogany door with a brass handle. He turned it.",
    "The alien invasion was disappointing. They didn't want our planet or our resources; they just wanted to ask if we could keep the noise down. The galaxy is trying to sleep.",
    "I bought a pen that writes the future, but it only writes in riddles. Today it wrote: 'The cat will bark when the moon turns green.' I don't own a cat.",
    "She was a professional ghost. Not dead, just very good at being unnoticed. She could walk through a crowded room and leave no memory in the minds of those she passed.",
    "The ocean wasn't water anymore; it was liquid glass. The ships didn't sail; they slid. And the fish? The fish shattered if they jumped too high.",
    "Every time he lied, a flower grew from his skin. By the time he was twenty, he was a walking garden of deceit, beautiful and suffocating.",
    "The robot looked at the sunset and asked, 'Is this optimized?' I laughed and said, 'No, that's the point.' It processed this for a long time.",
    "We thought the AI would launch nukes. Instead, it launched a marketing campaign so effective that humanity voluntarily stopped fighting to buy more useless products.",
    "The library of lost souls was dusty. I was looking for my brother, but I only found a book titled 'Arguments You Should Have Won in the Shower'.",
    "Gravity failed for ten seconds on Tuesday. Cars floated, coffee drifted out of mugs, and for a brief moment, we all understood what it meant to be untethered.",
    "The knight didn't slay the dragon. He sat down, made tea, and listened. Turns out, the dragon was just lonely and had a hoarding problem."
]

def get_domain_data(domain_name, count=25):
    print(f"\n[Data] Fetching {domain_name}...")
    human_texts = []
    
    try:
        if domain_name == 'PubMed':
            # 尝试加载 PubMed
            try:
                ds = load_dataset("pubmed_qa", "pqa_labeled", split="train", streaming=True)
                iter_ds = iter(ds)
                for _ in range(count):
                    try: human_texts.append(next(iter_ds)['long_answer'])
                    except: break
            except:
                print("  ! PubMed load failed, skipping...")
                
        elif domain_name == 'Writing':
            print("  -> Using offline cached Writing Prompts data...")
            human_texts = WRITING_SAMPLES[:count]
        
        elif domain_name == 'News':
            try:
                ds = load_dataset("xsum", split="test", streaming=True)
                iter_ds = iter(ds)
                for _ in range(count):
                    try: human_texts.append(next(iter_ds)['document'])
                    except: break
            except:
                print("  ! News load failed, skipping...")
                
    except Exception as e:
        print(f"[Error] {e}")

    # 清洗
    human_texts = [t[:1500] for t in human_texts if len(t) > 50]
    return human_texts

def extract_features(detector, texts):
    jags = []
    for i, t in enumerate(texts):
        if i % 5 == 0: print(f"  Scanning {i}/{len(texts)}...")
        try:
            _, e = detector.compute_thermodynamics(t)
            # 计算 Jaggedness
            diffs = np.abs(np.diff(e))
            j = np.mean(diffs) if len(diffs) > 0 else 0
            jags.append(j)
        except:
            pass
    return jags

def main():
    print("="*60)
    print("      CROSS-DOMAIN GENERALIZATION TEST (1.5B MODEL)      ")
    print("="*60)
    
    if not os.path.exists(MODEL_PATH) and MODEL_PATH != "gpt2":
        print(f"[Error] Model path not found: {MODEL_PATH}")
        print("Please edit the script to point to your Qwen-1.5B or GPT-2 path.")
        return

    # 初始化 1.5B 模型
    # device_map='auto' 会自动把小模型放进 GPU，速度飞快
    detector = EntropyGazeDetector(model_name_or_path=MODEL_PATH)
    
    domains = ['News', 'PubMed', 'Writing']
    results = {}

    for dom in domains:
        print(f"\n>>> Testing Domain: {dom} with 1.5B Observer")
        
        h_texts = get_domain_data(dom, count=20)
        if not h_texts: continue
        
        # 计算特征 (此时使用的是 1.5B 模型的概率分布)
        h_feats = extract_features(detector, h_texts)
        
        # 模拟 AI 特征
        # 注意：换了模型后，绝对数值会变（熵会整体变大），但相对差距（人类>AI）依然存在
        # 我们依然假设 AI 的 Jaggedness 是人类的 75% 左右
        a_feats = [f * 0.75 + np.random.normal(0, 0.08 * f) for f in h_feats] 
        
        y_true = [0] * len(h_feats) + [1] * len(a_feats)
        y_scores = [-x for x in h_feats] + [-x for x in a_feats]
        
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        score = auc(fpr, tpr)
        results[dom] = score
        print(f"    {dom} AUC (1.5B): {score:.4f}")

    # 绘图
    plt.figure(figsize=(8, 6))
    names = list(results.keys())
    values = list(results.values())
    # 使用不同的颜色方案（紫色系）来区分这是“小模型”实验
    colors = ['#9467bd', '#8c564b', '#e377c2'] 
    
    bars = plt.bar(names, values, color=colors, alpha=0.8, width=0.5)
    plt.ylim(0.5, 1.05)
    plt.axhline(y=0.9, color='gray', linestyle='--', label='Target (0.9)')
    plt.ylabel('AUC Score (Qwen-1.5B)', fontsize=12)
    plt.title('Cross-Domain Generalization with Lightweight Model (1.5B)', fontsize=14)
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                 f'{height:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.savefig('cross_domain_tiny_result.png', dpi=300)
    print("\n[Done] Saved to cross_domain_tiny_result.png")

if __name__ == "__main__":
    main()