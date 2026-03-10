import tensorflow as tf
from tensorflow.keras import layers, models, Input

class DreamGAN_Generator(models.Model):
    def __init__(self, hidden_dim, n_channels, max_seq_len):
        super(DreamGAN_Generator, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_channels = n_channels
        self.max_seq_len = max_seq_len
        
        # RNN-based Generator (TimeGAN style)
        self.gru1 = layers.GRU(hidden_dim, return_sequences=True)
        self.gru2 = layers.GRU(hidden_dim, return_sequences=True)
        self.linear = layers.Dense(n_channels, activation='tanh')
        
    def call(self, inputs):
        # inputs: Random Noise (Batch, Seq, Z_dim)
        x = self.gru1(inputs)
        x = self.gru2(x)
        output = self.linear(x)
        return output

class AuxiliaryClassifierDiscriminator(models.Model):
    """
    AC-TimeGAN Discriminator:
    Combines Real/Fake discrimination with an Auxiliary Classifier.
    
    Roles:
    1. Critic (Real/Fake): Guarantees Temporal/Structural realism (DMD/Microstates).
    2. Auxiliary Classifier (AC): Classifies the dream state (Emotion/Experience).
       This allows the model to handle class imbalance by learning the conditional distribution.
    """
    def __init__(self, hidden_dim, n_classes=3, n_microstates=4):
        super(AuxiliaryClassifierDiscriminator, self).__init__()
        
        # Shared Feature Extractor (Temporal Dynamics)
        self.gru = layers.GRU(hidden_dim, return_sequences=True)
        self.shared_dense = layers.Dense(hidden_dim, activation='leaky_relu')
        self.dropout = layers.Dropout(0.3) # Prevent overfitting/mode collapse
        
        # Head 1: Validity (Real/Fake) - Sequence Level
        self.validity_head = layers.Dense(1, activation='sigmoid')
        
        # Head 2: Auxiliary Classifier (The "AC" in AC-TimeGAN)
        # Classifies the sequence labels (e.g., Nightmare vs Lucid vs No Dream)
        self.pooling = layers.GlobalAveragePooling1D()
        self.auxiliary_classifier = layers.Dense(n_classes, activation='softmax') 
        
        # Head 3: Microstate Syntax (Generative Constraint)
        # Ensures the "grammar" (state transitions) is valid.
        self.syntax_head = layers.Dense(n_microstates, activation='softmax')
        
    def call(self, inputs):
        features = self.gru(inputs)
        features = self.shared_dense(features)
        features = self.dropout(features)
        
        # Outputs
        validity = self.validity_head(features)
        
        # Global features for AC
        global_features = self.pooling(features)
        class_pred = self.auxiliary_classifier(global_features)
        
        # Dense features for Syntax
        syntax_pred = self.syntax_head(features)
        
        return validity, class_pred, syntax_pred

def build_model(seq_len=256, n_channels=19, z_dim=10):
    """Builds and compiles the GAN."""
    generator = DreamGAN_Generator(hidden_dim=24, n_channels=n_channels, max_seq_len=seq_len)
    discriminator = AuxiliaryClassifierDiscriminator(hidden_dim=24)
    
    # Define optimization steps here...
    return generator, discriminator

# ==============================================================================
# PHASE 2: SEMANTIC-TIMEGAN (ZERO-SHOT EMOTION MAPPING)
# ==============================================================================

class SemanticDreamGAN_Generator(models.Model):
    def __init__(self, hidden_dim, n_channels, max_seq_len, embed_dim=128):
        super(SemanticDreamGAN_Generator, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_channels = n_channels
        self.max_seq_len = max_seq_len
        
        # We concatenate the semantic embedding into the temporal RNN layers.
        self.gru1 = layers.GRU(hidden_dim, return_sequences=True)
        self.gru2 = layers.GRU(hidden_dim, return_sequences=True)
        self.linear = layers.Dense(n_channels, activation='tanh')
        
    def call(self, inputs):
        # inputs is a tuple (z_noise, text_embedding)
        # z_noise: (Batch, Seq, Z_dim)
        # text_embedding: (Batch, Seq, embed_dim) # Tiled across time dimension
        z, semantic_cond = inputs
        x = tf.concat([z, semantic_cond], axis=-1)
        x = self.gru1(x)
        x = self.gru2(x)
        output = self.linear(x)
        return output

class SemanticTimeGAN_Discriminator(models.Model):
    """
    Semantic-TimeGAN Discriminator (Contrastive Projection Head)
    
    Instead of outputting probabilities [0.1, 0.8, 0.1] for fixed classes,
    it projects the physiological sequence into the exact same N-dimensional
    space as the NLP Text Encoders (e.g., BERT 128-D).
    """
    def __init__(self, hidden_dim, n_microstates=4, embed_dim=128):
        super(SemanticTimeGAN_Discriminator, self).__init__()
        
        self.gru = layers.GRU(hidden_dim, return_sequences=True)
        self.shared_dense = layers.Dense(hidden_dim, activation='leaky_relu')
        self.dropout = layers.Dropout(0.3)
        
        # 1. Real/Fake Validity
        self.validity_head = layers.Dense(1, activation='sigmoid')
        
        # 2. Semantic Projection Head (The massive Q1 Novelty)
        # Projects temporal physiology (EEG/ECG) directly into Semantic Text Space.
        self.pooling = layers.GlobalAveragePooling1D()
        self.semantic_projector = layers.Dense(embed_dim, activation='tanh') # Outputs 128-D vector
        
        # 3. Microstate Syntax
        self.syntax_head = layers.Dense(n_microstates, activation='softmax')
        
    def call(self, inputs):
        features = self.gru(inputs)
        features = self.shared_dense(features)
        features = self.dropout(features)
        
        # Validity
        validity = self.validity_head(features)
        
        # Semantic Extractor (Cosine mapping target)
        global_features = self.pooling(features)
        semantic_pred = self.semantic_projector(global_features) # (Batch, 128)
        
        # Syntax Check
        syntax_pred = self.syntax_head(features)
        
        return validity, semantic_pred, syntax_pred

def build_semantic_model(seq_len=256, n_channels=19, z_dim=32, embed_dim=128):
    """Builds the Semantic-Conditioned GAN."""
    generator = SemanticDreamGAN_Generator(hidden_dim=32, n_channels=n_channels, max_seq_len=seq_len, embed_dim=embed_dim)
    discriminator = SemanticTimeGAN_Discriminator(hidden_dim=32, embed_dim=embed_dim)
    return generator, discriminator

