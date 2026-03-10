import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch
from src.train import DreamGANTrainer

def debug_psd():
    config = {
        'data_root': r"c:\Users\Sachin.R\Downloads\Dream GAN",
        'n_channels': 19,
        'seq_len': 256,
        'z_dim': 32, 
        'dmd_rank': 5,
        'n_microstates': 4,
        'lr_gen': 0.001,
        'lr_disc': 0.001
    }
    
    print("Initializing Trainer...")
    trainer = DreamGANTrainer(config)
    
    # Build Models
    dummy_z = tf.zeros((1, 256, 32))
    dummy_x = tf.zeros((1, 256, 19))
    trainer.generator(dummy_z)
    trainer.discriminator(dummy_x)
    
    # Load Best Weights
    gen_weights = "results/best_generator.weights.h5"
    if os.path.exists(gen_weights):
        print(f"Loading Weights: {gen_weights}")
        trainer.generator.load_weights(gen_weights)
    else:
        print("Weights not found! Using random init (diagnosis will be limited).")

    # Get Data
    X_real, _ = trainer.prepare_data()
    
    # Generate Fake Data
    print("\n--- Diagnostic Statistics ---")
    z = tf.random.normal([len(X_real), 256, 32])
    X_fake = trainer.generator(z).numpy()
    
    # 1. Check Amplitude Statistics
    print(f"Real Data: Mean={np.mean(X_real):.6f}, Std={np.std(X_real):.6f}, Min={np.min(X_real):.6f}, Max={np.max(X_real):.6f}")
    print(f"Fake Data: Mean={np.mean(X_fake):.6f}, Std={np.std(X_fake):.6f}, Min={np.min(X_fake):.6f}, Max={np.max(X_fake):.6f}")
    
    # 2. Check for Tanh Saturation / Flatline
    if np.std(X_fake) < 0.01:
        print(" [CRITICAL] Fake data has near-zero variance! (Mode Collapse to Mean)")
    
    # 3. Computing PSD (Original vs De-normalized)
    print("\nComputng PSD...")
    # Helper for PSD
    def get_psd(data):
        f, p = welch(data.T, fs=250, nperseg=128) # Average across channels
        return f, np.mean(p, axis=0)
        
    psd_real_list = [get_psd(x)[1] for x in X_real]
    psd_fake_list = [get_psd(x)[1] for x in X_fake]
    
    psd_real_mean = np.mean(psd_real_list, axis=0)
    psd_fake_mean = np.mean(psd_fake_list, axis=0)
    freqs = get_psd(X_real[0])[0]
    
    # Plot 1: Raw Output Comparison
    plt.figure(figsize=(10, 6))
    plt.plot(freqs, psd_real_mean, label='Real EEG', color='blue')
    plt.plot(freqs, psd_fake_mean, label='Generated (Raw)', color='red', linestyle='--')
    plt.title('PSD Diagnosis: Raw Outputs')
    plt.yscale('log')
    plt.legend()
    plt.savefig("results/debug_psd_raw.png")
    print("Saved results/debug_psd_raw.png")
    
    # 4. Attempt Rescaling (Moment Matching)
    print("\nApplying Moment Matching Fix...")
    
    # Calculate moments
    mu_real = np.mean(X_real)
    sigma_real = np.std(X_real)
    mu_fake = np.mean(X_fake)
    sigma_fake = np.std(X_fake)
    
    # De-normalize Generator (Tanh -> Real Distribution)
    # Norm_Fake = (Fake - Mu_Fake) / Sigma_Fake
    # Target = Norm_Fake * Sigma_Real + Mu_Real
    X_fake_fixed = (X_fake - mu_fake) / sigma_fake * sigma_real + mu_real
    
    print(f"Moment Matching Applied:")
    print(f" - Real: u={mu_real:.3f}, s={sigma_real:.3f}")
    print(f" - Fake (Raw): u={mu_fake:.3f}, s={sigma_fake:.3f}")
    print(f" - Fake (Fix): u={np.mean(X_fake_fixed):.3f}, s={np.std(X_fake_fixed):.3f}")
    
    # Recompute PSD
    psd_fake_fixed_list = [get_psd(x)[1] for x in X_fake_fixed]
    psd_fake_fixed_mean = np.mean(psd_fake_fixed_list, axis=0)
    
    # Plot 3: Fixed Comparison
    plt.figure(figsize=(10, 6))
    plt.plot(freqs, psd_real_mean, label='Real EEG', color='blue', alpha=0.7)
    plt.plot(freqs, psd_fake_fixed_mean, label='Generated (Moment Matched)', color='red', linestyle='--', alpha=0.7)
    plt.fill_between(freqs, psd_real_mean, color='blue', alpha=0.1)
    
    plt.title('PSD Comparison (Fixed)')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power Spectral Density (V^2/Hz)')
    plt.legend()
    plt.xlim(0, 50)
    plt.yscale('log') # Log scale is crucial for EEG
    plt.grid(True, alpha=0.3)
    plt.savefig("results/debug_psd_fixed.png")
    print("Saved results/debug_psd_fixed.png")
    
if __name__ == "__main__":
    debug_psd()
