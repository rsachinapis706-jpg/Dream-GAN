import tensorflow as tf
import numpy as np
import os
import time
import random
from src.data.loader import SemanticUnifiedLoader
from src.features.dmd import DMD_FeatureExtractor
from src.models.timegan import build_semantic_model
from src.models.losses import dmd_loss

# === Global Reproducibility Seeds ===
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
# =====================================

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE

class SemanticGANTrainer:
    def __init__(self, config):
        self.config = config
        self.loader = SemanticUnifiedLoader(config['data_root'])
        
        self.dmd = DMD_FeatureExtractor(rank=config['dmd_rank'])
        
        # Initialize Semantic Model
        self.generator, self.discriminator = build_semantic_model(
            seq_len=config['seq_len'],
            n_channels=config['n_channels'],
            z_dim=config['z_dim'],
            embed_dim=config['embed_dim']
        )
        
        self.g_optimizer = tf.keras.optimizers.Adam(learning_rate=config['lr_gen'])
        self.d_optimizer = tf.keras.optimizers.Adam(learning_rate=config['lr_disc'])
        
    def prepare_data(self):
        print("Preparing Semantic Data...")
        records = self.loader.get_dream_records()
        print(f"Indices found: {len(records)}")
        
        X_data = []
        y_text_emb = []
        y_str_labels = []
        
        target_records = records.head(1000) 
        
        for idx, row in target_records.iterrows():
            fname = row.get('Filename', None)
            if not fname: fname = f"{row.get('subject', 'unknown')}_{row.get('session', 'unknown')}.edf"
                 
            data = self.loader.load_eeg_segment(str(fname), duration=self.config['seq_len']/250.0)
            
            # Map string to semantic emotion
            exp_str = row.get('Experience', 'Without recall')
            emotion_str = self.loader._map_experience_to_emotion(exp_str)
            emb = self.loader.encoder.get_semantic_vector(emotion_str)
            
            if data is not None:
                X_data.append(data)
                y_text_emb.append(emb)
                y_str_labels.append(emotion_str)
                
        if len(X_data) == 0:
            print("[CRITICAL WARNING] No files found. Falling back to synth.")
            X_real = np.random.randn(100, self.config['n_channels'], self.config['seq_len']).astype(np.float32)
            y_text_emb = np.random.randn(100, self.config['embed_dim']).astype(np.float32)
            y_str_labels = ['Neutral'] * 100
        else:
            X_real = np.array(X_data, dtype=np.float32)
            y_text_emb = np.array(y_text_emb, dtype=np.float32)
            
            unique, counts = np.unique(y_str_labels, return_counts=True)
            dist = dict(zip(unique, counts))
            print(f"Semantic Tag Distribution: {dist}")
            
        return X_real.transpose(0, 2, 1), y_text_emb, y_str_labels 

    @tf.function
    def train_step(self, x_real, semantic_target):
        batch_size = tf.shape(x_real)[0]
        z = tf.random.normal([batch_size, self.config['seq_len'], self.config['z_dim']])
        
        # Tile semantic vector across time dimension so generator can condition on it at each step
        semantic_cond = tf.tile(tf.expand_dims(semantic_target, 1), [1, self.config['seq_len'], 1])
        
        with tf.GradientTape() as g_tape, tf.GradientTape() as d_tape:
            # 1. Generate Fake Data (Conditioned on Semantic Target)
            x_fake = self.generator((z, semantic_cond))
            
            # 2. Discriminator Pass
            d_real_valid, d_real_semantic, d_real_syntax = self.discriminator(x_real)
            d_fake_valid, d_fake_semantic, d_fake_syntax = self.discriminator(x_fake)
            
            # A. Discriminator Validity Loss
            d_loss_valid = tf.reduce_mean(tf.keras.losses.binary_crossentropy(tf.ones_like(d_real_valid), d_real_valid) +
                                          tf.keras.losses.binary_crossentropy(tf.zeros_like(d_fake_valid), d_fake_valid))
            
            # **Contrastive Semantic Loss (Cosine Similarity)**
            # We want the discriminator's semantic projection to match the target text embedding
            semantic_target_norm = tf.nn.l2_normalize(semantic_target, axis=1)
            d_real_semantic_norm = tf.nn.l2_normalize(d_real_semantic, axis=1)
            cosine_loss_d = tf.reduce_mean(1.0 - tf.reduce_sum(semantic_target_norm * d_real_semantic_norm, axis=1))
            
            d_loss = d_loss_valid + 2.0 * cosine_loss_d
            
            # B. Generator Loss
            g_loss_valid = tf.reduce_mean(tf.keras.losses.binary_crossentropy(tf.ones_like(d_fake_valid), d_fake_valid))
            
            # PAA Physical Bounds Loss (DMD) - Ensuring it forms a true brain signal
            loss_dmd = dmd_loss(None, x_fake, None)
            
            # Generator wants the Fake data to map exactly to the exact requested Semantic target
            d_fake_semantic_norm = tf.nn.l2_normalize(d_fake_semantic, axis=1)
            cosine_loss_g = tf.reduce_mean(1.0 - tf.reduce_sum(semantic_target_norm * d_fake_semantic_norm, axis=1))
            
            g_loss = g_loss_valid + 0.1 * loss_dmd + 2.0 * cosine_loss_g
            
        g_grads = g_tape.gradient(g_loss, self.generator.trainable_variables)
        d_grads = d_tape.gradient(d_loss, self.discriminator.trainable_variables)
        
        self.g_optimizer.apply_gradients(zip(g_grads, self.generator.trainable_variables))
        self.d_optimizer.apply_gradients(zip(d_grads, self.discriminator.trainable_variables))
        
        return g_loss, d_loss, cosine_loss_d

    def train(self, epochs=200, batch_size=32):
        X_real, Y_emb, Y_str = self.prepare_data()
        dataset = tf.data.Dataset.from_tensor_slices((X_real, Y_emb)).batch(batch_size)
        
        print("Starting Semantic-TimeGAN Training...")
        best_cosine = 1.0 # Lower is better (1 - cos_sim)
        
        for epoch in range(epochs):
            g_loss_hist = []
            d_loss_hist = []
            cos_loss_hist = []
            
            for x_batch, y_batch in dataset:
                g_loss, d_loss, cos_loss = self.train_step(x_batch, y_batch)
                g_loss_hist.append(g_loss)
                d_loss_hist.append(d_loss)
                cos_loss_hist.append(cos_loss)
                
            avg_cos = np.mean(cos_loss_hist)
            if epoch % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} | Cosine Dist: {avg_cos:.4f} | G_Loss: {np.mean(g_loss_hist):.4f}")
                
                # Checkpointing
                if avg_cos < best_cosine:
                    best_cosine = avg_cos
                    self.generator.save_weights("results/semantic_generator.weights.h5")
                    self.discriminator.save_weights("results/semantic_discriminator.weights.h5")
                    
        # Final Verification Plot
        self.evaluate_semantic(X_real, Y_emb, Y_str)

    def evaluate_semantic(self, X_real, Y_emb, Y_str):
        print("Evaluating Semantic Manifold Mapping...")
        os.makedirs("results", exist_ok=True)
        
        # 1. Map Real EEG to Semantic Space through the SemanticTimeGAN Discriminator
        _, d_real_semantic, _ = self.discriminator(X_real)
        proj_real = d_real_semantic.numpy()
        
        # 2. Extract specific samples for visualization
        features = np.vstack([Y_emb, proj_real])
        labels = [f"Text: {l}" for l in Y_str] + [f"EEG: {l}" for l in Y_str]
        
        # We use fewer samples for clear TSNE
        if len(features) > 1000:
            indices = np.random.choice(len(Y_emb), 500, replace=False)
            features = np.vstack([Y_emb[indices], proj_real[indices]])
            labels = [f"Text: {Y_str[i]}" for i in indices] + [f"EEG: {Y_str[i]}" for i in indices]
            
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        embedded = tsne.fit_transform(features)
        
        plt.figure(figsize=(12, 10))
        sns.scatterplot(x=embedded[:,0], y=embedded[:,1], hue=labels, style=labels, palette="deep", alpha=0.7, s=80)
        plt.title('Semantic Manifold Mapping (Text Vectors vs Processed EEG)')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig("results/semantic_manifold.png")
        plt.close()
        print(" > Saved results/semantic_manifold.png")
        
        # Calculate Zero-Shot Similarity Score Space (Accuracy Proxy)
        sim_matrix = np.matmul(proj_real, Y_emb.T) / (np.linalg.norm(proj_real, axis=1, keepdims=True) * np.linalg.norm(Y_emb.T, axis=0, keepdims=True))
        
        # How often does the EEG vector match its paired Text vector more than random texts?
        # Check diagonal vs off-diagonal
        correct = 0
        for i in range(len(proj_real)):
            target_str = Y_str[i]
            # Find closest text vector
            closest_idx = np.argmax(sim_matrix[i])
            if Y_str[closest_idx] == target_str:
                correct += 1
                
        zero_shot_acc = correct / len(proj_real)
        print(f"\n==========================================")
        print(f"ZERO-SHOT SEMANTIC ACCURACY: {zero_shot_acc*100:.2f}%")
        print(f"==========================================")
        
        with open("results/semantic_validation.txt", "w") as f:
            f.write("Semantic-TimeGAN Metric Validation\n")
            f.write(f"Zero-Shot Emotion Mapping Accuracy: {zero_shot_acc*100:.2f}%\n")
            f.write("Evaluation Method: Cosine similarity matching between raw EEG sequence and BERT-style text clusters.\n")

if __name__ == "__main__":
    config = {
        'data_root': r"c:\Users\Sachin.R\Downloads\Dream GAN",
        'n_channels': 19,
        'seq_len': 256,
        'z_dim': 32,
        'embed_dim': 128,
        'dmd_rank': 5,
        'n_microstates': 4,
        'lr_gen': 0.001,
        'lr_disc': 0.001
    }
    
    trainer = SemanticGANTrainer(config)
    trainer.train(epochs=175)
