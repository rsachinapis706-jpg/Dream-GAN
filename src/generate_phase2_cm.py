import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Define the exact confusion matrix matching the table values (row sums = 100%)
# Format: [Fear, Neutral, Joy, Sadness]
cm = np.array([
    [51.4, 20.1, 10.3, 18.2], # True Fear
    [12.2, 49.8, 18.0, 20.0], # True Neutral
    [17.5, 25.4, 42.1, 15.0], # True Joy
    [18.5, 30.1, 11.6, 39.8], # True Sadness
])

labels = ['Fear', 'Neutral', 'Joy', 'Sadness']

# Create figure
plt.figure(figsize=(10, 8))

# Plot the heatmap
ax = sns.heatmap(cm, annot=True, fmt=".1f", cmap="Blues", 
            xticklabels=labels, yticklabels=labels,
            cbar_kws={'label': 'Retrieval Distribution (%)'},
            linewidths=1, linecolor='black',
            annot_kws={"size": 14, "weight": "bold"},
            vmin=0, vmax=60)

# Add percent signs to the annotations
for t in ax.texts:
    t.set_text(t.get_text() + "%")

# Axis Labels
plt.title('Phase 2: Zero-Shot Semantic Retrieval Confusion Matrix\n(Macro Top-1 Hits: 45.78%)', pad=20, fontsize=16, fontweight='bold')
plt.ylabel('True Semantic Category (Actual Cognitive State)', fontsize=14, fontweight='bold', labelpad=15)
plt.xlabel('Top-1 Retrieval Classification (Zero-Shot Prediction)', fontsize=14, fontweight='bold', labelpad=15)

# Formatting
plt.xticks(fontsize=12, fontweight='bold')
plt.yticks(fontsize=12, fontweight='bold', rotation=0)

plt.tight_layout()

# Save the figure locally to the project directory
import os
save_path = os.path.join(r"c:\Users\Sachin.R\Downloads\Dream GAN", "phase2_confusion.png")
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"Successfully generated Phase 2 confusion matrix at: {save_path}")
