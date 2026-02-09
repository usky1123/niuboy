# -*- coding: utf-8 -*-
import json
import numpy as np
import matplotlib.pyplot as plt
from dna_detectllm.detector import EntropyGazeDetector

# ================= CONFIGURATION =================
MODEL_PATH = "/root/autodl-tmp/models/LLM-Research/Meta-Llama-3-8B-Instruct"

# Files to compare
FILE_NORMAL = "Data/Collected data/GPT4_machine_test.json"
FILE_ATTACK = "Data/Text_attack/GPT4_machine_test_replace.json"
FILE_HUMAN  = "Data/Collected data/xsum_human.json"
# =================================================

def load_first_text(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            data = json.load(f)
            # Find any key with 'text'
            for key in data.keys():
                if 'text' in key:
                    return data[key][0] # Just take the first sample
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
    return None

def main():
    print(f"[RTX 5090] Initializing Detector...")
    detector = EntropyGazeDetector(model_name_or_path=MODEL_PATH)
    
    print("Loading samples...")
    txt_normal = load_first_text(FILE_NORMAL)
    txt_attack = load_first_text(FILE_ATTACK)
    txt_human  = load_first_text(FILE_HUMAN)
    
    if not (txt_normal and txt_attack and txt_human):
        print("Error: Could not load all three samples.")
        return

    print("Computing Thermodynamics...")
    _, e_normal = detector.compute_thermodynamics(txt_normal)
    _, e_attack = detector.compute_thermodynamics(txt_attack)
    _, e_human  = detector.compute_thermodynamics(txt_human)
    
    # Calculate Variances
    v_normal = np.var(e_normal)
    v_attack = np.var(e_attack)
    v_human  = np.var(e_human)
    
    print(f"\n--- Variance Analysis ---")
    print(f"1. Normal AI (GPT-4): {v_normal:.4f} (Smooth)")
    print(f"2. Human (XSum)     : {v_human:.4f}  (Natural)")
    print(f"3. Attacked AI      : {v_attack:.4f} (Chaos!)")
    
    if v_attack > v_human:
        print("\n[Conclusion] The attack increased variance BEYOND human levels.")
        print("             This explains why AUC dropped below 0.5.")

    # Plotting
    plt.figure(figsize=(12, 8))
    
    # Plot 1: Normal AI
    plt.subplot(3, 1, 1)
    plt.plot(e_normal, color='#1f77b4', lw=1.5)
    plt.title(f"Normal GPT-4 (Var: {v_normal:.2f}) - Adiabatic/Smooth")
    plt.ylim(0, 15)
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Human
    plt.subplot(3, 1, 2)
    plt.plot(e_human, color='#2ca02c', lw=1.5)
    plt.title(f"Human XSum (Var: {v_human:.2f}) - Dissipative/Natural")
    plt.ylim(0, 15)
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Attacked AI
    plt.subplot(3, 1, 3)
    plt.plot(e_attack, color='#d62728', lw=1.5)
    plt.title(f"Attacked GPT-4 (Var: {v_attack:.2f}) - Artificial Turbulence")
    plt.ylim(0, 15)
    plt.xlabel("Token Index")
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('attack_analysis_plot.png', dpi=300)
    print("\n[Success] Analysis plot saved to: attack_analysis_plot.png")

if __name__ == "__main__":
    main()