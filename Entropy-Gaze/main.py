import json
import os
import numpy as np
import matplotlib.pyplot as plt
from dna_detectllm.detector import EntropyGazeDetector

def run_5090_experiment():
    # Correct path based on your previous success
    model_path = "/root/autodl-tmp/models/LLM-Research/Meta-Llama-3-8B-Instruct"
    
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return

    # Initialize Detector
    detector = EntropyGazeDetector(model_name_or_path=model_path)

    # Define Datasets
    test_cases = [
        {"name": "AI_GPT4", "path": "Data/Collected data/GPT4_machine_test.json", "key": "machine_text"},
        {"name": "Human_XSum", "path": "Data/Collected data/xsum_human.json", "key": "human_text"}
    ]

    print("\n" + "="*50)
    print("Starting Entropy-Gaze Analysis (RTX 5090 Mode)")
    print("="*50)

    for case in test_cases:
        # Handle relative paths safely
        full_path = os.path.join(os.getcwd(), case['path'])
        if not os.path.exists(full_path):
            print(f"Warning: Skipping {case['name']}, file not found at {full_path}")
            continue

        print(f"\nProcessing Group: {case['name']} ...")
        # Use errors='ignore' to prevent utf-8 decoding issues in data files
        with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
            texts = json.load(f)[case['key']][:50] 

        vars, fractures = [], []
        for i, text in enumerate(texts):
            m, energies = detector.compute_thermodynamics(text)
            vars.append(m['energy_variance'])
            fractures.append(m['fracture_points'])
            
            # Save visualization for the first sample
            if i == 0:
                plt.figure(figsize=(10, 4))
                plt.plot(energies, color='blue', alpha=0.7)
                plt.title(f"Energy Flow: {case['name']}")
                plt.xlabel("Token Index")
                plt.ylabel("Self-Information (Energy)")
                plt.savefig(f"spectrogram_{case['name']}.png")
                plt.close()

        print(f"Results for {case['name']}:")
        print(f"   - Average Variance: {np.mean(vars):.4f}")
        print(f"   - Average Fracture Points: {np.mean(fractures):.2f}")

if __name__ == "__main__":
    run_5090_experiment()