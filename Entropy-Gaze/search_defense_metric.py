# -*- coding: utf-8 -*-
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from dna_detectllm.detector import EntropyGazeDetector

# ================= CONFIGURATION =================
MODEL_PATH = "/root/autodl-tmp/models/LLM-Research/Meta-Llama-3-8B-Instruct"
FILE_HUMAN  = "Data/Collected data/xsum_human.json"
FILE_ATTACK = "Data/Text_attack/GPT4_machine_test_replace.json"
# =================================================

def load_data(filepath, limit=50):
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            data = json.load(f)
            for key in data.keys():
                if 'text' in key: return data[key][:limit]
    except: pass
    return []

def compute_advanced_metrics(energies):
    e = np.array(energies)
    # 1. Variance (Total Energy Fluctuation)
    var = np.var(e)
    
    # 2. Fracture Count (k=2.0, lowered threshold)
    thresh = np.mean(e) + 2.0 * np.std(e)
    frac = np.sum(e > thresh)
    
    # 3. Jaggedness (Mean Absolute Difference / First Derivative)
    # Measures how "jumpy" the sequence is locally
    diffs = np.abs(np.diff(e))
    jaggedness = np.mean(diffs) if len(diffs) > 0 else 0
    
    # 4. Local Volatility (Mean of rolling std, window=5)
    if len(e) > 5:
        local_vol = np.mean([np.std(e[i:i+5]) for i in range(len(e)-5)])
    else:
        local_vol = 0
        
    return var, frac, jaggedness, local_vol

def main():
    print("[RTX 5090] Searching for Robust Defense Metrics...")
    detector = EntropyGazeDetector(model_name_or_path=MODEL_PATH)
    
    texts_human = load_data(FILE_HUMAN)
    texts_attack = load_data(FILE_ATTACK)
    
    print(f"[Processing] {len(texts_human)} Human vs {len(texts_attack)} Attack samples...")

    # Store metrics
    metrics_human = {'var':[], 'frac':[], 'jag':[], 'vol':[]}
    metrics_attack = {'var':[], 'frac':[], 'jag':[], 'vol':[]}

    for t in texts_human:
        _, e = detector.compute_thermodynamics(t)
        v, f, j, vol = compute_advanced_metrics(e)
        metrics_human['var'].append(v)
        metrics_human['frac'].append(f)
        metrics_human['jag'].append(j)
        metrics_human['vol'].append(vol)

    for t in texts_attack:
        _, e = detector.compute_thermodynamics(t)
        v, f, j, vol = compute_advanced_metrics(e)
        metrics_attack['var'].append(v)
        metrics_attack['frac'].append(f)
        metrics_attack['jag'].append(j)
        metrics_attack['vol'].append(vol)

    # Compare AUCs for all candidates
    # Label: 0=Human, 1=Attack
    y_true = [0]*len(texts_human) + [1]*len(texts_attack)
    
    print("\n" + "="*50)
    print("       DEFENSE METRIC LEADERBOARD       ")
    print("="*50)
    
    best_auc = 0
    best_metric = ""
    best_fpr, best_tpr = None, None

    # Test each metric
    metric_names = {
        'var': 'Variance (Baseline)',
        'frac': 'Fracture Count (k=2.0)',
        'jag': 'Jaggedness (1st Derivative)',
        'vol': 'Local Volatility (Rolling Std)'
    }

    for key, name in metric_names.items():
        scores = metrics_human[key] + metrics_attack[key]
        fpr, tpr, _ = roc_curve(y_true, scores)
        score = auc(fpr, tpr)
        
        # If AUC < 0.5, it means the relationship is inverted (Human > Attack). 
        # Flip it to make it comparable (Defense strength).
        if score < 0.5:
            score = 1 - score
            
        print(f"Candidate: {name:<30} | AUC: {score:.4f}")
        
        if score > best_auc:
            best_auc = score
            best_metric = name
            best_fpr, best_tpr = fpr, tpr

    print("-" * 50)
    print(f"? WINNER: {best_metric} (AUC={best_auc:.4f})")
    
    if best_auc > 0.75:
        print(" Conclusion: We found a robust fingerprint against attacks!")
    else:
        print(" Conclusion: Still struggling. Consider combining metrics.")

    # Plot the winner
    plt.figure(figsize=(8, 6))
    plt.plot(best_fpr, best_tpr, color='darkorange', lw=2, label=f'{best_metric} (AUC={best_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.title(f'Best Defense Strategy: {best_metric}')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.savefig('best_defense_roc.png')
    print("[Plot] Saved to 'best_defense_roc.png'")

if __name__ == "__main__":
    main()