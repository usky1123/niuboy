# -*- coding: utf-8 -*-
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from dna_detectllm.detector import EntropyGazeDetector

# ================= CONFIGURATION =================
# 1. Model Path (RTX 5090)
MODEL_PATH = "/root/autodl-tmp/models/LLM-Research/Meta-Llama-3-8B-Instruct"

# 2. Base Directory (Root of your data folder)
# Assuming your folder structure starts inside 'Data/'
BASE_DIR = "Data/"

# 3. AI Models to Compare (Mapped to your provided tree structure)
# Format: "Display Name": "Relative Path from BASE_DIR"
AI_MODELS = {
    # --- Mainstream Models (Collected data) ---
    "GPT-4": "Collected data/GPT4_machine_test.json",
    "Claude": "Collected data/Claude_machine_test.json",
    "Gemini": "Collected data/Gemini_machine_test.json",
    
    # --- Benchmarks ---
    "DetectRL (Multi-LLM)": "DetectRL/DetectRL_multillm_machine_test.json",
    "M4 (Mixed)": "M4/M4_machine_test.json",
    "RealDet": "RealDet/RealDet_machine_test.json",
    
    # --- Adversarial Attacks (Robustness Test) ---
    # Testing if the detector fails when text is modified (e.g., replaced or inserted)
    "GPT-4 (Attack: Replace)": "Text_attack/GPT4_machine_test_replace.json",
    "Claude (Attack: Insert)": "Text_attack/Claude_machine_test_insert.json"
}

# 4. Human Baseline Data (Negative Class)
# You have xsum, arxiv, wp. xsum is good for general summary style.
HUMAN_DATA = "Collected data/xsum_human.json"
# =================================================

def load_data(filename):
    path = os.path.join(BASE_DIR, filename)
    if not os.path.exists(path):
        print(f"[Warning] File not found: {path}")
        return []
    
    # Load JSON safely
    try:
        with open(path, "r", encoding='utf-8', errors='ignore') as f:
            data = json.load(f)
    except Exception as e:
        print(f"[Error] Failed to load {filename}: {e}")
        return []
        
    # Smart key detection logic to handle different dataset formats
    keys = list(data.keys())
    
    # 1. Try finding specific machine text keys
    target_key = next((k for k in keys if "machine" in k and "text" in k), None) 
    
    # 2. Try finding model text keys (common in DetectRL/M4)
    if not target_key:
        target_key = next((k for k in keys if "model" in k and "text" in k), None)
        
    # 3. Fallback to any key containing 'text'
    if not target_key:
        target_key = next((k for k in keys if "text" in k), None)
        
    if target_key:
        # Limit to 50 samples for speed benchmark
        # You can increase this number if you want more accurate results
        print(f"  [Load] Found key '{target_key}' in {filename}")
        return data[target_key][:50]
    else:
        print(f"[Warning] No valid text key found in {filename}. Available keys: {keys}")
        return []

def main():
    print("="*60)
    print(f"[Info] Initializing Detector on RTX 5090...")
    print(f"[Info] Model: {MODEL_PATH}")
    print("="*60)
    
    detector = EntropyGazeDetector(model_name_or_path=MODEL_PATH)
    
    # 1. Load Human Baseline
    print(f"\n[Info] Loading Human Baseline: {HUMAN_DATA}...")
    human_texts = load_data(HUMAN_DATA)
    if not human_texts:
        print("[Error] Human data not found. Please check paths. Aborting.")
        return
    
    print(f"[Info] Computing Entropy Flow for {len(human_texts)} Human texts...")
    human_energies = []
    for t in human_texts:
        try:
            _, e = detector.compute_thermodynamics(t)
            human_energies.append(np.var(e))
        except Exception as e:
            print(f"  [Skip] Error processing human text: {e}")
            
    # 2. Benchmark AI Models
    plt.figure(figsize=(12, 10))
    # Define a rich color palette for many lines
    colors = [
        '#d62728', '#2ca02c', '#1f77b4', # GPT4, Claude, Gemini (RGB)
        '#9467bd', '#8c564b', '#e377c2', # Benchmarks (Purple, Brown, Pink)
        '#7f7f7f', '#bcbd22', '#17becf'  # Attacks (Gray, Olive, Cyan)
    ]
    
    print("\n--- Starting Multi-Model Benchmark ---")
    
    for i, (model_name, filename) in enumerate(AI_MODELS.items()):
        texts = load_data(filename)
        if not texts:
            continue
            
        print(f"[Processing] {model_name} ({len(texts)} samples)...")
        
        ai_energies = []
        for t in texts:
            try:
                _, e = detector.compute_thermodynamics(t)
                ai_energies.append(np.var(e))
            except Exception as e:
                # Skip samples that are too short or cause errors
                continue
            
        if not ai_energies:
            print(f"  [Warning] No valid energies computed for {model_name}")
            continue

        # Compute ROC-AUC
        # y_true: AI=0, Human=1 (Hypothesis: Human variance is higher)
        # We perform minimum length matching to avoid array size mismatch
        min_len = min(len(ai_energies), len(human_energies))
        
        # Slicing to ensure balanced evaluation if counts differ
        current_ai = ai_energies[:min_len]
        current_human = human_energies[:min_len]
        
        y_true = [0] * len(current_ai) + [1] * len(current_human)
        y_scores = current_ai + current_human
        
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        
        print(f"   -> {model_name} AUC: {roc_auc:.4f}")
        
        # Plot Curve
        plt.plot(fpr, tpr, lw=2, color=colors[i % len(colors)], 
                 label=f'{model_name} (AUC = {roc_auc:.2f})')

    # 3. Finalize Plot
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Thermodynamic Consistency: Comprehensive Model Comparison')
    plt.legend(loc="lower right", fontsize='small')
    plt.grid(True, alpha=0.3)
    
    save_path = 'roc_comparison_comprehensive.png'
    plt.savefig(save_path, dpi=300)
    print(f"\n[Success] Comprehensive plot saved to: {save_path}")

if __name__ == "__main__":
    main()