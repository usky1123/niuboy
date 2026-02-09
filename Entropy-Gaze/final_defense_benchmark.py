# -*- coding: utf-8 -*-
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from dna_detectllm.detector import EntropyGazeDetector

# ================= CONFIGURATION =================
MODEL_PATH = "/root/autodl-tmp/models/LLM-Research/Meta-Llama-3-8B-Instruct"

# Compare Human vs. The "Hardest" Enemy (Attacked AI)
FILE_HUMAN  = "Data/Collected data/xsum_human.json"
FILE_ATTACK = "Data/Text_attack/GPT4_machine_test_replace.json"
# =================================================

def load_data(filepath, limit=50):
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            data = json.load(f)
            # Smart key search
            for key in data.keys():
                if 'text' in key: return data[key][:limit]
    except: pass
    return []

def main():
    print("="*50)
    print("[RTX 5090] Final Defense Benchmark")
    print("Goal: Distinguish 'Attacked AI' from 'Human'")
    print("="*50)
    
    detector = EntropyGazeDetector(model_name_or_path=MODEL_PATH)
    
    texts_human = load_data(FILE_HUMAN)
    texts_attack = load_data(FILE_ATTACK)
    
    if not texts_human or not texts_attack:
        print("[Error] Data loading failed.")
        return

    print(f"[Processing] {len(texts_human)} Human vs {len(texts_attack)} Attacked AI samples...")

    # metric lists
    var_human, frac_human = [], []
    var_attack, frac_attack = [], []

    # 1. Compute metrics for Human
    print("  -> Scanning Human texts...")
    for t in texts_human:
        m, _ = detector.compute_thermodynamics(t)
        var_human.append(m['energy_variance'])
        frac_human.append(m['fracture_points'])

    # 2. Compute metrics for Attacked AI
    print("  -> Scanning Attacked AI texts...")
    for t in texts_attack:
        m, _ = detector.compute_thermodynamics(t)
        var_attack.append(m['energy_variance'])
        frac_attack.append(m['fracture_points'])

    # 3. Compare AUCs
    y_true = [0] * len(texts_human) + [1] * len(texts_attack) # 0=Human, 1=Attack (Positive Class)

    # Method A: Variance (Expecting Failure)
    y_score_var = var_human + var_attack
    fpr_v, tpr_v, _ = roc_curve(y_true, y_score_var)
    auc_var = auc(fpr_v, tpr_v)

    # Method B: Fracture Count (Expecting Success)
    y_score_frac = frac_human + frac_attack
    fpr_f, tpr_f, _ = roc_curve(y_true, y_score_frac)
    auc_frac = auc(fpr_f, tpr_f)

    print("\n" + "="*50)
    print("      ? FINAL RESULTS ?      ")
    print("="*50)
    print(f"Method A (Variance) AUC : {auc_var:.4f}")
    print(f"   -> Interpretation: {('Failed (Close to 0.5)' if abs(auc_var-0.5)<0.15 else 'Distinguishable')}")
    
    print(f"Method B (Fractures) AUC: {auc_frac:.4f}")
    print(f"   -> Interpretation: {('Success!' if auc_frac > 0.6 else 'Needs improvement')}")
    
    # Save comparison plot
    plt.figure(figsize=(8, 6))
    plt.plot(fpr_v, tpr_v, linestyle='--', label=f'Variance (AUC={auc_var:.2f}) - Failed')
    plt.plot(fpr_f, tpr_f, linewidth=3, color='red', label=f'Fracture Count (AUC={auc_frac:.2f}) - Defense')
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    plt.title('Defense Strategy: Variance vs. Fracture Count')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.savefig('final_defense_auc.png')
    print("\n[Plot] Saved to 'final_defense_auc.png'")

if __name__ == "__main__":
    main()