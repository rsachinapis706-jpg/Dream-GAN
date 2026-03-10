import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics.pairwise import cosine_similarity
from src.data.deed_loader import DEEDLoader
from src.models.timegan import build_semantic_model

# Force CPU output for stable inference plotting
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

def generate_retrieval_cm():
    print("Loading DEED Dataset and Anchor Embeddings...")
    data_dir = r"c:\Users\Sachin.R\Downloads\Dream GAN\dream_eeg\Dream EEG with emotion labels"
    loader = DEEDLoader(data_dir, seq_len=3000)
    X_real, Y_text, Y_emb = loader.load_data()
    
    # Defining the exact 6 classes in the DEED dataset
    class_names = ['No Recall', 'Negative', 'Rel. Negative', 'Neutral', 'Rel. Positive', 'Positive']
    emotion_texts = [
        'I did not experience any dream or have no dream recall.',
        'I had a negative affective dream experience.',
        'I had a relatively negative affective dream experience.',
        'I had a neutral affective dream experience.',
        'I had a relatively positive affective dream experience.',
        'I had a positive affective dream experience.'
    ]
    
    # We use SentenceTransformer to convert these 6 base classes into strict mathematical "Anchors"
    if loader.nlp_model:
        anchor_embs = loader.nlp_model.encode(emotion_texts)
    else:
        print("ERROR: sentence-transformers not found.")
        return
        
    print("Loading Trained Semantic-TimeGAN Weights...")
    generator, discriminator = build_semantic_model(seq_len=3000, n_channels=6, z_dim=32, embed_dim=384)
    weights_g = r'c:\Users\Sachin.R\Downloads\Dream GAN\results\deed_semantic_generator.weights.h5'
    weights_d = r'c:\Users\Sachin.R\Downloads\Dream GAN\results\deed_semantic_discriminator.weights.h5'
    
    # Build models in Keras 3 with dummy pass
    generator([tf.zeros([1, 3000, 32]), tf.zeros([1, 3000, 384])])
    discriminator(tf.zeros([1, 3000, 6]))
    
    generator.load_weights(weights_g)
    discriminator.load_weights(weights_d)
    
    print("Executing Zero-Shot Semantic Retrieval across all 533 Real brainwaves...")
    
    # The Discriminator acts as the BCI retrieval agent: it takes the True Biological Brainwave
    # and projects it into the 384-D Semantic Space to decode the emotion.
    _, proj_real, _ = discriminator.predict(X_real, verbose=0)
    
    y_true = []
    y_pred = []
    
    # 3. Use the exact unique embedding extraction logic from `train_deed.py`
    unique_Y_emb, unique_indices = np.unique(Y_emb, axis=0, return_index=True)
    sim_matrix_unique = cosine_similarity(proj_real, unique_Y_emb)
    
    # Track exactly which of the 6 unique embeddings maps to which text label
    unique_labels = [Y_text[idx] for idx in unique_indices]
    
    # Map the unique labels to their standard logical sorting order for the Confusion Matrix
    # to maintain our ['No Recall', 'Negative', 'Rel. Negative', 'Neutral', 'Rel. Positive', 'Positive'] array
    ordered_indices_map = []
    for txt in class_names:
        if txt in unique_labels:
            ordered_indices_map.append(unique_labels.index(txt))
        else:
            ordered_indices_map.append(-1) # Fallback if missing
    
    for i in range(len(X_real)):
        true_text = Y_text[i]
        true_idx = emotion_texts.index(true_text)
        y_true.append(true_idx)
        
        # Rank the predictions against the unique semantic anchors
        ranked_indices = np.argsort(sim_matrix_unique[i])[::-1]
        
        # The AI's predicted target index (which of the unique embeddings it picked)
        pred_unique_idx = ranked_indices[0]
        
        # Map the unique index back to the standard 0-5 class sorting
        pred_text = unique_labels[pred_unique_idx]
        pred_idx = emotion_texts.index(pred_text)
        
        y_pred.append(pred_idx)
    
        
    print("Plotting Q1 Retrieval Confusion Matrix...")
    cm_raw = confusion_matrix(y_true, y_pred, labels=range(6))
    
    # Convert absolute counts to row-wise percentages for intuitive reading
    cm_perc = cm_raw.astype('float') / cm_raw.sum(axis=1)[:, np.newaxis]
    cm_perc = np.nan_to_num(cm_perc) * 100
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_perc, annot=True, fmt='.1f', cmap='magma', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Retrieval Probability (%)'})
    
    plt.title("Zero-Shot Emotion Retrieval Confusion Matrix (Real DEED n=533)", fontsize=14, pad=15)
    plt.xlabel("Predicted Emotion (AI Deduced directly from Real Brainwave)", fontsize=12, labelpad=10)
    plt.ylabel("Ground Truth Target (Original Text Prompt)", fontsize=12, labelpad=10)
    plt.xticks(rotation=45, ha='right', fontsize=11)
    plt.yticks(rotation=0, fontsize=11)
    
    os.makedirs(r"c:\Users\Sachin.R\Downloads\Dream GAN\results\deed_figures", exist_ok=True)
    plt.tight_layout()
    plt.savefig(r"c:\Users\Sachin.R\Downloads\Dream GAN\results\deed_figures\retrieval_confusion_matrix.png", dpi=300, bbox_inches='tight')
    plt.close()

    hits1 = np.mean(np.array(y_true) == np.array(y_pred)) * 100
    print(f"Verified Final Hits@1 Aggregate: {hits1:.2f}%")
    print("DONE! Plot saved to results/deed_figures/retrieval_confusion_matrix.png")

if __name__ == "__main__":
    generate_retrieval_cm()
