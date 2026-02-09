# -*- coding: utf-8 -*-
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from dna_detectllm.detector import EntropyGazeDetector

# ================= 配置区域 =================
# 1. 现有的 Llama-3 (作为基准)
MODEL_A_PATH = "/root/autodl-tmp/models/LLM-Research/Meta-Llama-3-8B-Instruct" 

# 2. 自动下载 Qwen2.5 (作为对比)
# Qwen 是目前最强的开源模型之一，作为验证者非常有说服力
DOWNLOAD_DIR = "/root/autodl-tmp/models/Qwen"
TARGET_MODEL_ID = "qwen/Qwen2.5-7B-Instruct"

DATA_FILE = "Data/Collected data/xsum_human.json"
# ===========================================

def auto_download_model():
    print("="*60)
    print("      AUTO DOWNLOADING QWEN2.5      ")
    print("="*60)
    try:
        from modelscope import snapshot_download
    except ImportError:
        print("[Installing] modelscope library...")
        os.system("pip install modelscope")
        from modelscope import snapshot_download
    
    print(f"[Info] Downloading {TARGET_MODEL_ID} from ModelScope...")
    try:
        model_dir = snapshot_download(TARGET_MODEL_ID, cache_dir=DOWNLOAD_DIR)
        print(f"[Success] Model downloaded to: {model_dir}")
        return model_dir
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
    print(f"\n[Info] Loading Detector: {label}...")
    try:
        # 释放显存，防止两个模型同时加载炸显存
        import torch
        torch.cuda.empty_cache()
        
        detector = EntropyGazeDetector(model_name_or_path=model_path)
    except Exception as e:
        print(f"[Error] Failed to load {label}: {e}")
        return None

    variances = []
    print(f"  -> Scanning {len(texts)} samples with {label}...")
    for t in texts:
        _, e = detector.compute_thermodynamics(t)
        variances.append(np.var(e))
    
    # 手动删除模型以释放显存给下一个
    del detector
    return variances

def main():
    # 1. Check/Download Qwen
    model_b_path = auto_download_model()
    if not model_b_path:
        print("Skipping experiment due to download failure.")
        return

    # 2. Load Data
    texts = load_data(DATA_FILE, limit=50)
    if not texts:
        print("Data not found.")
        return

    # 3. Run Llama-3 (Benchmark)
    print("\n--- Phase 1: Running Llama-3 ---")
    var_a = get_metrics(MODEL_A_PATH, texts, "Llama-3")
    
    # 4. Run Qwen (Validator)
    print("\n--- Phase 2: Running Qwen2.5 ---")
    var_b = get_metrics(model_b_path, texts, "Qwen2.5")

    if var_a is None or var_b is None:
        print("Experiment aborted.")
        return

    # 5. Correlation Analysis
    corr, _ = pearsonr(var_a, var_b)
    
    print("\n" + "="*60)
    print(f"PEARSON CORRELATION: {corr:.4f}")
    print("="*60)
    
    if corr > 0.6:
        print(" SUCCESS: The laws of physics hold true across models!")
        print("   (Llama-3 and Qwen saw the same thermodynamic pattern)")
    else:
        print(" Result: Low correlation.")

    # 6. Plot
    plt.figure(figsize=(8, 6))
    plt.scatter(var_a, var_b, alpha=0.7, color='teal')
    plt.title(f'Detector Universality: Llama-3 vs Qwen (r={corr:.2f})')
    plt.xlabel('Llama-3 Entropy Variance')
    plt.ylabel('Qwen2.5 Entropy Variance')
    plt.grid(True, alpha=0.3)
    plt.savefig('detector_generalization.png')
    print("[Plot] Saved to 'detector_generalization.png'")

if __name__ == "__main__":
    main()