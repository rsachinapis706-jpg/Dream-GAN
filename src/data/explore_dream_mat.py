import os
import glob
import re

target_dir = r"c:\Users\Sachin.R\Downloads\Dream GAN\dream_eeg\Dream EEG with emotion labels"

# Get a list of all .mat files
mat_files = glob.glob(os.path.join(target_dir, "*.mat"))

labels = set()
for f in mat_files:
    basename = os.path.basename(f)
    # Format seems to be like: G_S0021_M3_E0_R4_nan_raw_ref.mat
    parts = basename.split('_')
    if len(parts) >= 6:
        label = parts[3] # E0, E1, E2, E3, E4, E5?
        stage = parts[5] # nan, N1, N2, W, REM
        labels.add((label, stage))

print(f"Total files: {len(mat_files)}")
print("Unique Emotion Codes (Label, Sleep Stage):")
for l in sorted(labels):
    print(l)
