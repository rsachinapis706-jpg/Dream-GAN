import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

class MicrostateExtractor:
    """
    Extracts EEG Microstates (The 'Atoms of Thought').
    Used to enforce 'Syntactic' realism in the generative model.
    
    Standard Maps:
    A: Auditory/Phonological
    B: Visual
    C: Salience/Autonomic
    D: Attention/Working Memory
    """
    
    def __init__(self, n_states: int = 4):
        self.n_states = n_states
        self.maps = None # The microstate prototype maps (n_states, n_channels)
        self.kmeans = None
        
    def fit(self, X_epochs):
        """
        Clustering to find microstate prototypes.
        
        Args:
            X_epochs: List of EEG epochs or single large array (n_channels, n_times)
                      Must be GFP-peak data usually, but we'll simplify for now.
        """
        # Flatten data: (n_channels, total_time) -> (total_time, n_channels)
        # We cluster the *topographies* at each time point.
        
        if isinstance(X_epochs, list):
            data = np.concatenate(X_epochs, axis=1).T
        else:
            data = X_epochs.T
            
        # Normalize topographies (Gfp=1)
        data = data / np.std(data, axis=1, keepdims=True)
        
        # Train K-Means
        # Note: Microstates ignores polarity, but standard K-Means respects it. 
        # For a rigorous implementation we would use Modified K-Means. 
        # For this prototype, standard K-Means on absolute values or simple K-Means is a starting point.
        # We will use standard K-Means for robustness in this phase.
        self.kmeans = KMeans(n_clusters=self.n_states, random_state=42)
        self.kmeans.fit(data)
        self.maps = self.kmeans.cluster_centers_
        
    def predict_sequence(self, X):
        """
        Converts raw EEG into a sequence of microstate labels (Symbolic Sequence).
        
        Args:
            X: EEG data (n_channels, n_times)
            
        Returns:
            labels: (n_times,) array of 0..3 representing active microstate.
        """
        if self.kmeans is None:
            raise RuntimeError("Model not fitted.")
            
        data = X.T
        data = data / np.std(data, axis=1, keepdims=True)
        
        labels = self.kmeans.predict(data)
        return labels
    
    def get_transition_matrix(self, labels):
        """
        Computes the Markov transition matrix from the label sequence.
        """
        n_states = self.n_states
        M = np.zeros((n_states, n_states))
        
        for (i, j) in zip(labels[:-1], labels[1:]):
            M[i, j] += 1
            
        # Normalize
        row_sums = M.sum(axis=1, keepdims=True)
        M = np.divide(M, row_sums, out=np.zeros_like(M), where=row_sums!=0)
        
        return M

if __name__ == "__main__":
    # Test
    # Simulate 19-channel EEG
    data = np.random.randn(19, 1000)
    model = MicrostateExtractor(n_states=4)
    model.fit(data)
    seq = model.predict_sequence(data)
    trans_mat = model.get_transition_matrix(seq)
    
    print("Microstate Sequence Head:", seq[:20])
    print("Transition Probability Matrix:\n", np.round(trans_mat, 2))
