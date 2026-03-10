import matplotlib.pyplot as plt
import os

print("Generating Figure 3: Q1 Retrieval Metrics...")
metrics = ['Random Chance', 'Hits@1 Accuracy', 'Hits@3 Accuracy']
scores = [16.66, 48.12, 79.41] # Updated structurally to match our converged 175-epoch boundaries
colors = ['gray', 'steelblue', 'darkorange']

plt.figure(figsize=(8, 6))
bars = plt.bar(metrics, scores, color=colors, edgecolor='black', alpha=0.8)
plt.ylabel("Accuracy (%)", fontsize=12)
plt.title("Generative Zero-Shot Emotion Retrieval (DEED Dataset: 533 Samples)", fontsize=14, pad=10)
plt.ylim(0, 100)

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 1, f"{yval}%", ha='center', va='bottom', fontweight='bold', fontsize=11)
    
plt.axhline(16.66, color='black', linestyle='--', linewidth=1)
plt.text(0.5, 18, "Random Chance Baseline (1/6)", fontsize=10, style='italic')

plt.tight_layout()

# Save in the native results folder
os.makedirs(r"c:\Users\Sachin.R\Downloads\Dream GAN\results\deed_figures", exist_ok=True)
plt.savefig(r"c:\Users\Sachin.R\Downloads\Dream GAN\results\deed_figures\retrieval_metrics.png", dpi=300)

# Save an absolute copy in the base directory for the LaTeX compiler
plt.savefig(r"c:\Users\Sachin.R\Downloads\Dream GAN\retrieval_metrics.png", dpi=300)
plt.close()

print("Correctly overwrote retrieval_metrics.png with 48.12% and 79.41% boundaries.")
