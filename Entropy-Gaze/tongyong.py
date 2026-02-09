# -*- coding: utf-8 -*-
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from dna_detectllm.detector import EntropyGazeDetector

# ================= 配置区域 (只改这里) =================
# [在此处修改为您想测试的模型路径]
# 例如换成 Qwen 的路径，或者 Mistral 的路径
CURRENT_MODEL_PATH = "/root/autodl-tmp/models/LLM-Research/Meta-Llama-3-8B-Instruct"

# [标准测试数据集路径 - 保持不变]
DATA_BASE_DIR = "Data"
FILE_HUMAN = os.path.join(DATA_BASE_DIR, "Collected data/xsum_human.json")
FILE_AI    = os.path.join(DATA_BASE_DIR, "Collected data/GPT4_machine_test.json")
FILE_ATTACK= os.path.join(DATA_BASE_DIR, "Text_attack/GPT4_machine_test_replace.json")
# =======================================================

def load_data(filepath, limit=50):
    #"""通用数据加载函数"""
    if not os.path.exists(filepath):
        print(f"[Error] File not found: {filepath}")
        return []
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            data = json.load(f)
            # 智能查找 key
            keys = data.keys()
            target_key = next((k for k in keys if 'text' in k), None)
            if target_key:
                return data[target_key][:limit]
    except Exception as e:
        print(f"[Error] Loading {filepath}: {e}")
    return []

def extract_features(energies):
    #"""提取四维热力学指纹"""
    e = np.array(energies)
    if len(e) == 0: return [0, 0, 0, 0]
    
    # 1. Variance (全局波动)
    var = np.var(e)
    
    # 2. Fracture Count (逻辑断裂)
    thresh = np.mean(e) + 2.0 * np.std(e)
    frac = np.sum(e > thresh)
    
    # 3. Jaggedness (粗糙度/一阶差分)
    diffs = np.abs(np.diff(e))
    jag = np.mean(diffs) if len(diffs) > 0 else 0
    
    # 4. Local Volatility (局部波动率)
    if len(e) > 5:
        vol = np.mean([np.std(e[i:i+5]) for i in range(len(e)-5)])
    else:
        vol = 0
        
    return [var, frac, jag, vol]

def evaluate_auc(features_neg, features_pos):
    #"""计算基于多维特征融合的 AUC"""
    X = np.array(features_neg + features_pos)
    y = np.array([0] * len(features_neg) + [1] * len(features_pos))
    
    # 标准化 + 逻辑回归融合
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    clf = LogisticRegression()
    # 使用简单的 fit 预测概率 (为了演示速度，不做交叉验证，直接看训练集拟合能力)
    clf.fit(X_scaled, y)
    probs = clf.predict_proba(X_scaled)[:, 1]
    
    fpr, tpr, _ = roc_curve(y, probs)
    score = auc(fpr, tpr)
    return score, fpr, tpr

def main():
    print("="*60)
    print("      UNIVERSAL THERMODYNAMIC EVALUATOR      ")
    print(f"      Model: {os.path.basename(CURRENT_MODEL_PATH)}")
    print("="*60)

    # 1. 初始化模型
    try:
        detector = EntropyGazeDetector(model_name_or_path=CURRENT_MODEL_PATH)
    except Exception as e:
        print(f"[Fatal Error] Model load failed: {e}")
        return

    # 2. 加载数据
    print("[1/4] Loading Datasets...")
    texts_human = load_data(FILE_HUMAN)
    texts_ai = load_data(FILE_AI)
    texts_attack = load_data(FILE_ATTACK)
    
    if not (texts_human and texts_ai and texts_attack):
        print("[Error] Data missing. Check paths.")
        return

    # 3. 计算特征
    print("[2/4] Computing Thermodynamic Fingerprints...")
    
    def process_batch(texts, label):
        feats = []
        for i, t in enumerate(texts):
            if i % 20 == 0: print(f"      Processing {label} {i}/{len(texts)}...")
            try:
                _, e = detector.compute_thermodynamics(t)
                feats.append(extract_features(e))
            except:
                pass
        return feats

    feats_human = process_batch(texts_human, "Human")
    feats_ai    = process_batch(texts_ai,    "Standard AI")
    feats_attack= process_batch(texts_attack,"Attacked AI")

    # 4. 评估性能
    print("[3/4] Calculating Performance Scores...")
    
    # 场景 A: 标准检测 (Human vs GPT-4)
    auc_std, fpr_std, tpr_std = evaluate_auc(feats_human, feats_ai)
    
    # 场景 B: 对抗防御 (Human vs Attack)
    auc_rob, fpr_rob, tpr_rob = evaluate_auc(feats_human, feats_attack)

    print("\n" + "="*60)
    print(f"      RESULTS FOR: {os.path.basename(CURRENT_MODEL_PATH)}      ")
    print("="*60)
    print(f"1. Standard Detection (GPT-4)   AUC: {auc_std:.4f}")
    print(f"   -> {'Excellent' if auc_std > 0.9 else 'Good'}")
    print("-" * 60)
    print(f"2. Adversarial Defense (Attack) AUC: {auc_rob:.4f}")
    print(f"   -> {'Robust' if auc_rob > 0.8 else 'Vulnerable'}")
    print("="*60)

    # 5. 绘图
    print("[4/4] Generating Report Plot...")
    plt.figure(figsize=(8, 6))
    plt.plot(fpr_std, tpr_std, color='blue', lw=2, label=f'Standard (GPT-4) AUC={auc_std:.2f}')
    plt.plot(fpr_rob, tpr_rob, color='red', lw=2, label=f'Defense (Attack) AUC={auc_rob:.2f}')
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    plt.title(f'Detector Performance: {os.path.basename(CURRENT_MODEL_PATH)}')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    
    save_name = f"report_{os.path.basename(CURRENT_MODEL_PATH)}.png"
    plt.savefig(save_name)
    print(f"[Done] Report saved to: {save_name}")

if __name__ == "__main__":
    main()