# -*- coding: utf-8 -*-
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
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

def extract_features(energies):
    e = np.array(energies)
    if len(e) == 0: return [0,0,0,0]
    
    # Feature 1: Variance
    var = np.var(e)
    
    # Feature 2: Fracture Count (k=2.0)
    thresh = np.mean(e) + 2.0 * np.std(e)
    frac = np.sum(e > thresh)
    
    # Feature 3: Jaggedness (1st Derivative)
    diffs = np.abs(np.diff(e))
    jag = np.mean(diffs) if len(diffs) > 0 else 0
    
    # Feature 4: Local Volatility
    if len(e) > 5:
        vol = np.mean([np.std(e[i:i+5]) for i in range(len(e)-5)])
    else:
        vol = 0
        
    return [var, frac, jag, vol]

def main():
    print("[RTX 5090] Initializing Multi-Dimensional Defense System...")
    detector = EntropyGazeDetector(model_name_or_path=MODEL_PATH)
    
    # 1. Load Data
    texts_human = load_data(FILE_HUMAN)
    texts_attack = load_data(FILE_ATTACK)
    print(f"[Data] Loaded {len(texts_human)} Human and {len(texts_attack)} Attack samples.")

    # 2. Feature Extraction
    print("[Processing] Extracting thermodynamic features...")
    X = []
    y = [] # 0 = Human, 1 = Attack

    for t in texts_human:
        _, e = detector.compute_thermodynamics(t)
        X.append(extract_features(e))
        y.append(0)

    for t in texts_attack:
        _, e = detector.compute_thermodynamics(t)
        X.append(extract_features(e))
        y.append(1)

    X = np.array(X)
    y = np.array(y)
    
    # Normalize features (Crucial for PCA/Logistic Regression)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 3. Model Training (Cross-Validation)
    print("\n" + "="*50)
    print("      COMBINED DEFENSE RESULTS      ")
    print("="*50)
    
    # Classifier A: Logistic Regression (Linear)
    clf_lr = LogisticRegression()
    scores_lr = cross_val_score(clf_lr, X_scaled, y, cv=5, scoring='roc_auc')
    print(f"Logistic Regression AUC: {scores_lr.mean():.4f} (+/- {scores_lr.std() * 2:.2f})")

    # Classifier B: Random Forest (Non-linear)
    clf_rf = RandomForestClassifier(n_estimators=100, random_state=42)
    scores_rf = cross_val_score(clf_rf, X_scaled, y, cv=5, scoring='roc_auc')
    print(f"Random Forest AUC      : {scores_rf.mean():.4f} (+/- {scores_rf.std() * 2:.2f})")
    
    if scores_rf.mean() > 0.75:
        print(" SUCCESS: Multi-feature fusion effectively detects attacks!")
    else:
        print(" NOTE: Even combined features struggle. This is a strong scientific result about attack potency.")

    # 4. Visualization (PCA)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    plt.figure(figsize=(10, 8))
    # Plot Human
    plt.scatter(X_pca[y==0, 0], X_pca[y==0, 1], 
                color='green', label='Human', alpha=0.7, edgecolors='k')
    # Plot Attack
    plt.scatter(X_pca[y==1, 0], X_pca[y==1, 1], 
                color='red', label='Attacked AI', alpha=0.7, edgecolors='k', marker='^')
    
    plt.title(f'Thermodynamic Feature Space (PCA)\nCombined AUC: {max(scores_lr.mean(), scores_rf.mean()):.2f}')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('defense_pca_plot.png')
    print("[Plot] PCA Visualization saved to 'defense_pca_plot.png'")

if __name__ == "__main__":
    main()