# -*- coding: utf-8 -*-
import json
import numpy as np
from dna_detectllm.detector import EntropyGazeDetector

# ================= CONFIGURATION =================
MODEL_PATH = "/root/autodl-tmp/models/LLM-Research/Meta-Llama-3-8B-Instruct"
FILE_NORMAL = "Data/Collected data/GPT4_machine_test.json"
FILE_ATTACK = "Data/Text_attack/GPT4_machine_test_replace.json"
FILE_HUMAN  = "Data/Collected data/xsum_human.json"
# =================================================

def load_first_text(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            data = json.load(f)
            for key in data.keys():
                if 'text' in key: return data[key][0]
    except Exception as e:
        print(f"[Error] loading {filepath}: {e}")
    return None

def main():
    print("[RTX 5090] Diagnostic Scan Started...")
    detector = EntropyGazeDetector(model_name_or_path=MODEL_PATH)
    
    txt_normal = load_first_text(FILE_NORMAL)
    txt_attack = load_first_text(FILE_ATTACK)
    txt_human  = load_first_text(FILE_HUMAN)

    if not (txt_normal and txt_attack and txt_human):
        print("[Error] Could not load all samples.")
        return

    # Compute Thermodynamics
    m_normal, _ = detector.compute_thermodynamics(txt_normal)
    m_attack, _ = detector.compute_thermodynamics(txt_attack)
    m_human, _  = detector.compute_thermodynamics(txt_human)
    
    print("\n" + "="*40)
    print("      LOGICAL FRACTURE ANALYSIS      ")
    print("="*40)
    print(f"1. Normal AI (GPT-4) Fractures : {m_normal['fracture_points']}")
    print(f"2. Human (XSum) Fractures      : {m_human['fracture_points']}")
    print(f"3. Attacked AI Fractures       : {m_attack['fracture_points']}")
    print("-" * 40)

    # Automatic Judgment
    if m_attack['fracture_points'] > m_human['fracture_points']:
        print("[SUCCESS] Defense Effective!")
        print("   Although variance is similar, the attack sample has significantly")
        print("   more 'logical fractures' (unnatural spikes).")
        print("   Conclusion: Fracture Count is a valid secondary filter.")
    else:
        print("[WARNING] Attack is sophisticated.")
        print("   The attack mimics both variance and fracture count of humans.")

if __name__ == "__main__":
    main()