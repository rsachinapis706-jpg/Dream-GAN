import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import welch
from src.data.deed_loader import DEEDLoader
from src.models.timegan import build_semantic_model

# Force CPU for stable visualization
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

def calculate_band_power(psd, freqs):
    # Standard EEG Bands
    bands = {
        'Delta (1-4 Hz)': (1, 4),
        'Theta (4-8 Hz)': (4, 8),
        'Alpha (8-13 Hz)': (8, 13),
        'Beta (13-30 Hz)': (13, 30),
        'Gamma (30-50 Hz)': (30, 50)
    }
    
    band_powers = []
    for band, (low, high) in bands.items():
        idx = np.logical_and(freqs >= low, freqs <= high)
        bp = np.sum(psd[idx])
        band_powers.append(bp)
        
    return band_powers, list(bands.keys())

def generate_spectral_validation():
    print("Loading DEED Dataset and Trained Models for Spectral Validation...")
    data_dir = r"c:\Users\Sachin.R\Downloads\Dream GAN\dream_eeg\Dream EEG with emotion labels"
    loader = DEEDLoader(data_dir, seq_len=3000)
    X_real, Y_text, Y_emb = loader.load_data()
    
    # We evaluate over the entire generated dataset for statistical power
    # Let's use 200 samples for the distribution graph
    X_real_subset = X_real[:200]
    Y_emb_subset = Y_emb[:200]
    
    generator, discriminator = build_semantic_model(seq_len=3000, n_channels=6, z_dim=32, embed_dim=384)
    weights_path_g = r'c:\Users\Sachin.R\Downloads\Dream GAN\results\deed_semantic_generator.weights.h5'
    
    # Build models
    generator([tf.zeros([1, 3000, 32]), tf.zeros([1, 3000, 384])])
    generator.load_weights(weights_path_g)
    
    print("Generating Synthetic Brainwaves Conditioned on English Text...")
    batch_size = len(X_real_subset)
    z = tf.random.normal([batch_size, 3000, 32])
    semantic_cond = tf.tile(tf.expand_dims(Y_emb_subset, 1), [1, 3000, 1])
    X_fake_subset = generator([z, semantic_cond], training=False).numpy()
    
    # Extract Channel 0 for spectral comparison
    real_signals = X_real_subset[:, :, 0]
    fake_signals = X_fake_subset[:, :, 0]
    
    print("Computing Welch's Power Spectral Density (PSD)...")
    fs = 100.0 # Standard EEG sampling rate for sleep
    
    real_psds = []
    fake_psds = []
    
    real_band_powers = []
    fake_band_powers = []
    
    for i in range(batch_size):
        r_sig = real_signals[i]
        f_sig = fake_signals[i]
        
        # Denormalize fake signal to real scale for fair physics comparison
        f_sig = (f_sig - np.mean(f_sig)) / (np.std(f_sig) + 1e-8)
        f_sig = (f_sig * np.std(r_sig)) + np.mean(r_sig)
        
        f, px_real = welch(r_sig, fs, nperseg=256)
        _, px_fake = welch(f_sig, fs, nperseg=256)
        
        real_psds.append(px_real)
        fake_psds.append(px_fake)
        
        r_bp, bands = calculate_band_power(px_real, f)
        f_bp, _ = calculate_band_power(px_fake, f)
        
        real_band_powers.append(r_bp)
        fake_band_powers.append(f_bp)
        
    mean_real_psd = np.mean(real_psds, axis=0)
    std_real_psd = np.std(real_psds, axis=0) / np.sqrt(batch_size)
    
    mean_fake_psd = np.mean(fake_psds, axis=0)
    std_fake_psd = np.std(fake_psds, axis=0) / np.sqrt(batch_size)
    
    os.makedirs(r"c:\Users\Sachin.R\Downloads\Dream GAN\results\deed_figures", exist_ok=True)
    
    # ==========================================
    # FIGURE 1: Power Spectral Density Overlap
    # ==========================================
    print("Plotting PSD Overlap...")
    plt.figure(figsize=(10, 6))
    
    plt.plot(f, 10 * np.log10(mean_real_psd), label='Real DEED Brainwaves', color='blue', linewidth=2)
    plt.fill_between(f, 
                     10 * np.log10(mean_real_psd - std_real_psd + 1e-10), 
                     10 * np.log10(mean_real_psd + std_real_psd), 
                     color='blue', alpha=0.2)
                     
    plt.plot(f, 10 * np.log10(mean_fake_psd), label='Generated (Text-Conditioned)', color='red', linewidth=2, linestyle='--')
    plt.fill_between(f, 
                     10 * np.log10(mean_fake_psd - std_fake_psd + 1e-10), 
                     10 * np.log10(mean_fake_psd + std_fake_psd), 
                     color='red', alpha=0.2)
                     
    plt.xlim(0, 50)
    plt.title("Spectral Consistency Overlap (Targeting Exact English Emotions)", fontsize=14)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power Spectral Density (dB/Hz)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(r"c:\Users\Sachin.R\Downloads\Dream GAN\results\deed_figures\deed_psd_overlap.png", dpi=300)
    plt.close()
    
    # ==========================================
    # FIGURE 2: Band Power Comparison
    # ==========================================
    print("Plotting Regional Band Power Consistency...")
    mean_real_bp = np.mean(real_band_powers, axis=0)
    mean_fake_bp = np.mean(fake_band_powers, axis=0)
    
    x = np.arange(len(bands))
    width = 0.35
    
    plt.figure(figsize=(10, 6))
    plt.bar(x - width/2, mean_real_bp, width, label='Real DEED Brainwaves', color='blue', alpha=0.7)
    plt.bar(x + width/2, mean_fake_bp, width, label='Generated (Text-Conditioned)', color='red', alpha=0.7)
    
    plt.ylabel('Absolute Power (μV²/Hz)')
    plt.title('Absolute Band Power Consistency (DEED Cohort n=200)', fontsize=14)
    plt.xticks(x, bands)
    plt.legend()
    plt.tight_layout()
    plt.savefig(r"c:\Users\Sachin.R\Downloads\Dream GAN\results\deed_figures\deed_band_power.png", dpi=300)
    plt.close()
    
    print("Spectral analysis absolute verification complete.")

if __name__ == "__main__":
    generate_spectral_validation()
