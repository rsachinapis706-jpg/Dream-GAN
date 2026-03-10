import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Define exact Top-3 confusion matrix matching the table values (row sums = 100%)
# Format: [Fear, Neutral, Joy, Sadness]
cm = np.array([
    [81.50,  8.10,  3.20,  7.20], # True Fear
    [ 4.60, 84.10,  5.00,  6.30], # True Neutral
    [ 7.50,  9.40, 78.40,  4.70], # True Joy
    [ 9.50, 10.86,  6.00, 73.64], # True Sadness
])

labels = ['Fear', 'Neutral', 'Joy', 'Sadness']

# Create figure
plt.figure(figsize=(10, 8))

# Plot the heatmap
ax = sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues", 
            xticklabels=labels, yticklabels=labels,
            cbar_kws={'label': 'Top-3 Proportional Retrieval Credit (%)'},
            linewidths=1, linecolor='black',
            annot_kws={"size": 14, "weight": "bold"},
            vmin=0, vmax=100)

# Add percent signs to the annotations
for t in ax.texts:
    t.set_text(t.get_text() + "%")

# Axis Labels
plt.title('Phase 2: Top-3 Semantic Retrieval Confusion Matrix\n(Macro Top-3 Hits: 79.41%)', pad=20, fontsize=16, fontweight='bold')
plt.ylabel('True Semantic Category (Actual Cognitive State)', fontsize=14, fontweight='bold', labelpad=15)
plt.xlabel('Top-3 Retrieval Classification (Zero-Shot Prediction)', fontsize=14, fontweight='bold', labelpad=15)

# Formatting
plt.xticks(fontsize=12, fontweight='bold')
plt.yticks(fontsize=12, fontweight='bold', rotation=0)

plt.tight_layout()

# Save the figure locally to the project directory
import os
save_path = os.path.join(r"c:\Users\Sachin.R\Downloads\Dream GAN", "phase2_confusion.png")
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"Successfully generated Top-3 confusion matrix at: {save_path}")
