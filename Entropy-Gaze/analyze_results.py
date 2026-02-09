import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from dna_detectllm.detector import EntropyGazeDetector

# 1. Load Model (Using the path confirmed to work on your 5090)
model_path = "/root/autodl-tmp/models/LLM-Research/Meta-Llama-3-8B-Instruct"
detector = EntropyGazeDetector(model_name_or_path=model_path)

print("Starting deep scan to find optimal thresholds...")

# 2. Load Data (with error ignoring to prevent encoding crashes)
with open("Data/Collected data/GPT4_machine_test.json", "r", encoding='utf-8', errors='ignore') as f:
    ai_texts = json.load(f)["machine_text"][:50]
with open("Data/Collected data/xsum_human.json", "r", encoding='utf-8', errors='ignore') as f:
    human_texts = json.load(f)["human_text"][:50]

# 3. Batch Compute Energies
def get_energies(texts, label):
    all_energies = []
    print(f"Computing energy flow for {label}...")
    for i, t in enumerate(texts):
        _, e = detector.compute_thermodynamics(t)
        all_energies.append(e)
    return all_energies

ai_energies_list = get_energies(ai_texts, "AI (GPT-4)")
human_energies_list = get_energies(human_texts, "Human (XSum)")

# 4. Grid Search for Best 'k' (Fracture Threshold)
best_k = 0
best_diff = 0
print("\n--- Calibrating Fracture Threshold (k) ---")
for k in np.arange(1.5, 4.0, 0.5):
    # Calculate average fracture points for this k
    ai_frac = np.mean([np.sum(e > (np.mean(e) + k * np.std(e))) for e in ai_energies_list])
    hu_frac = np.mean([np.sum(e > (np.mean(e) + k * np.std(e))) for e in human_energies_list])
    
    diff = abs(hu_frac - ai_frac)
    print(f"k={k:.1f} -> AI: {ai_frac:.2f} vs Human: {hu_frac:.2f} (Diff: {diff:.2f})")
    
    if diff > best_diff:
        best_diff = diff
        best_k = k

print(f"\n[Recommendation] Best Threshold: k = {best_k}")

# 5. Calculate ROC-AUC based on Variance
# Hypothesis: Human variance (1) > AI variance (0)
y_true = [0] * len(ai_texts) + [1] * len(human_texts)
y_scores = [np.var(e) for e in ai_energies_list] + [np.var(e) for e in human_energies_list]

fpr, tpr, thresholds = roc_curve(y_true, y_scores)
roc_auc = auc(fpr, tpr)

print(f"\n[Result] ROC-AUC Score based on Energy Variance: {roc_auc:.4f}")
if roc_auc > 0.7:
    print("   -> Significant distinction found! Good for paper.")
else:
    print("   -> Distinction is weak. Consider testing longer texts.")

# 6. Save ROC Plot
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title(f'ROC: Thermodynamic Variance (Best k={best_k})')
plt.legend(loc="lower right")
plt.savefig('roc_curve_final.png')
print("Graph saved as 'roc_curve_final.png'")