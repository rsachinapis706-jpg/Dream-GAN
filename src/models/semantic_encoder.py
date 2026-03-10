import numpy as np
import tensorflow as tf

class SemanticTextEncoder:
    """
    Simulated NLP Semantic Encoder.
    Since we don't have raw NLP mentation text in the open dataset CSVs,
    we generate synthetic semantic vectors representing emotional states
    (e.g., as if extracted by BERT or CLIP).
    
    This provides Ground Truth semantic vectors for cross-modal training.
    """
    def __init__(self, embed_dim=128):
        self.embed_dim = embed_dim
        
        # Define predefined cluster centers for specific semantics
        # Dimensionality: embed_dim
        
        # We define 3 main semantic clusters
        # 1. Nightmare / High Arousal (Fear, Falling)
        self.nightmare_center = np.random.randn(embed_dim) * 0.5 + 1.0
        
        # 2. Positive / Joy (Flying, Lucid)
        self.joy_center = np.random.randn(embed_dim) * 0.5 - 1.0
        
        # 3. Neutral / No Recall (Darkness, Nothing)
        self.neutral_center = np.zeros(embed_dim)
        
    def get_semantic_vector(self, label: str) -> np.ndarray:
        """
        Given a text label or class, return a continuous 'text embedding'.
        In reality, this would be: `return bert.encode(text_report)`
        """
        noise = np.random.randn(self.embed_dim) * 0.1 # Add slight variance to simulate text diversity
        
        if label == 'Nightmare':
            return self.nightmare_center + noise
        elif label == 'Joy':
            return self.joy_center + noise
        elif label == 'Neutral' or label == 'No Recall':
            return self.neutral_center + noise
        else:
            return np.random.randn(self.embed_dim)

    def get_batch_embeddings(self, labels: list) -> np.ndarray:
        return np.array([self.get_semantic_vector(l) for l in labels], dtype=np.float32)

if __name__ == "__main__":
    encoder = SemanticTextEncoder()
    v1 = encoder.get_semantic_vector("Nightmare")
    v2 = encoder.get_semantic_vector("Nightmare")
    v3 = encoder.get_semantic_vector("Neutral")
    
    # Cosine similarity check to ensure it works
    sim1_2 = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    sim1_3 = np.dot(v1, v3) / (np.linalg.norm(v1) * np.linalg.norm(v3))
    
    print(f"Similarity (Within Class Nightmare): {sim1_2:.4f}")
    print(f"Similarity (Cross Class Nightmare vs Neutral): {sim1_3:.4f}")
