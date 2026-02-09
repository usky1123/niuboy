# -*- coding: utf-8 -*-
import os
import gc
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from dna_detectllm.detector import EntropyGazeDetector

# ================= 配置区域 =================
# 1. 您的基准模型 (Llama-3-8B)
BASE_MODEL_PATH = "/root/autodl-tmp/models/LLM-Research/Meta-Llama-3-8B-Instruct"

# 2. 自动下载 Qwen2.5-1.5B (轻量级，仅 3GB 左右，下载极快)
# 这足以作为“异构模型”来验证物理规律的普适性
TARGET_MODEL_ID = "qwen/Qwen2.5-1.5B-Instruct"

DOWNLOAD_DIR = "/root/autodl-tmp/models"
DATA_FILE = "Data/Collected data/xsum_human.json"
# ===========================================

def auto_download(model_id):
    print("="*60)
    print(f"   AUTO DOWNLOADING TINY MODEL: {model_id}   ")
    print("="*60)
    try:
        from modelscope import snapshot_download
    except ImportError:
        os.system("pip install modelscope")
        from modelscope import snapshot_download
    
    print(f"[Download] Fetching from ModelScope (Fast)...")
    try:
        path = snapshot_download(model_id, cache_dir=DOWNLOAD_DIR)
        print(f"[Success] Path: {path}")
        return path
    except Exception as e:
        print(f"[Error] Download failed: {e}")
        return None

def load_data(filepath, limit=50):
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            data = json.load(f)
            for key in data.keys():
                if 'text' in key: return data[key][:limit]
    except: pass
    return []

def get_metrics(model_path, texts, label):
    print(f"\n[Running] Detector: {label}...")
    
    # 显存清理大法
    gc.collect()
    torch.cuda.empty_cache()
    
    try:
        # 1.5B 模型非常小，加载毫无压力
        detector = EntropyGazeDetector(model_name_or_path=model_path)
    except Exception as e:
        print(f"[Error] Load failed: {e}")
        return None

    variances = []
    for i, t in enumerate(texts):
        if i % 10 == 0: print(f"  -> Scanning {i}/{len(texts)}...")
        try:
            _, e = detector.compute_thermodynamics(t)
            variances.append(np.var(e))
        except:
            variances.append(0) # 容错
    
    del detector
    gc.collect()
    torch.cuda.empty_cache()
    return variances

def main():
    print("="*60)
    print("      DETECTOR AGNOSTICISM TEST (TINY MODEL)      ")
    print("="*60)
    
    texts = load_data(DATA_FILE, limit=50)
    if not texts: return

    # 1. Run Base Model (Llama-3)
    print("\n--- Phase 1: Reference Model (Llama-3) ---")
    base_vars = get_metrics(BASE_MODEL_PATH, texts, "Llama-3-8B")
    if not base_vars: return

    # 2. Download & Run Tiny Target Model
    print("\n--- Phase 2: Validator Model (Qwen-1.5B) ---")
    target_path = auto_download(TARGET_MODEL_ID)
    if not target_path: return
    
    target_vars = get_metrics(target_path, texts, "Qwen2.5-1.5B")
    
    if not target_vars: return

    # 3. Correlation Analysis
    # 过滤掉可能的无效数据
    valid_indices = [i for i in range(len(base_vars)) if base_vars[i] > 0 and target_vars[i] > 0]
    clean_base = [base_vars[i] for i in valid_indices]
    clean_target = [target_vars[i] for i in valid_indices]

    corr, _ = pearsonr(clean_base, clean_target)
    
    print("\n" + "="*60)
    print(f"PEARSON CORRELATION: {corr:.4f}")
    print("="*60)
    
    if corr > 0.5:
        print(" SUCCESS: Even a tiny model (1.5B) sees the same thermodynamic pattern!")
        print("   This strongly supports the 'Detector Agnosticism' claim.")
    
    # 4. Plot
    plt.figure(figsize=(8, 6))
    plt.scatter(clean_base, clean_target, alpha=0.7, c='teal')
    
    # 拟合线
    m, b = np.polyfit(clean_base, clean_target, 1)
    plt.plot(clean_base, np.array(clean_base)*m + b, color='orange', linestyle='--')
    
    plt.title(f"Detector Consistency: Llama-3 (8B) vs Qwen (1.5B)\nr = {corr:.2f}")
    plt.xlabel("Llama-3 Variance")
    plt.ylabel("Qwen-1.5B Variance")
    plt.grid(True, alpha=0.3)
    plt.savefig('universality_check_tiny.png')
    print("[Plot] Saved to 'universality_check_tiny.png'")

if __name__ == "__main__":
    main()