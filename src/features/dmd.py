import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class DMD_FeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Dynamic Mode Decomposition (DMD) Feature Extractor.
    Extracts physics-informed dynamical modes from EEG data.
    
    References:
    - User's Concept (maths.pdf): X' = AX
    - Tu et al. (2014) on Exact DMD.
    """
    
    def __init__(self, rank: int = 0):
        """
        Args:
            rank: Rank for truncation (SVD). 0 means no truncation.
        """
        self.rank = rank
        self.eigenvalues = None
        self.modes = None
        self.amplitudes = None
        
    def fit(self, X, y=None):
        """
        Computes the DMD of the input data X.
        
        Args:
            X: Input data of shape (n_channels, n_timesteps)
               Note: Single epoch processing.
        """
        # 1. Create Snapshot Matrices
        # X1 = x[0, 1, ..., m-1]
        # X2 = x[1, 2, ..., m]
        X1 = X[:, :-1]
        X2 = X[:, 1:]
        
        # 2. SVD of X1
        U, S, Vh = np.linalg.svd(X1, full_matrices=False)
        
        if self.rank > 0:
            U = U[:, :self.rank]
            S = S[:self.rank]
            Vh = Vh[:self.rank, :]
            
        # 3. Compute A_tilde (Koopman operator approximation)
        # A_tilde = U.T @ X2 @ V @ S_inv
        S_inv = np.diag(1.0 / S)
        A_tilde = U.T @ X2 @ Vh.T @ S_inv
        
        # 4. Eigendecomposition of A_tilde
        eigvals, eigvecs = np.linalg.eig(A_tilde)
        
        # 5. Compute Exact DMD Modes
        # Phi = X2 @ V @ S_inv @ W
        self.modes = X2 @ Vh.T @ S_inv @ eigvecs
        self.eigenvalues = eigvals
        
        return self
        
    def get_features(self):
        """
        Returns the extracted physics-informed features.
        
        Returns:
            Dictionary containing:
            - eigenvalues: Stability/Oscillation info (Real/Imag parts)
            - modes: Spatial patterns of the modes
        """
        if self.eigenvalues is None:
            raise RuntimeError("Model not fitted yet.")
            
        return {
            "eigenvalues_real": self.eigenvalues.real,
            "eigenvalues_imag": self.eigenvalues.imag,
            "abs_eigenvalues": np.abs(self.eigenvalues) # Stability metric (|lambda| ~ 1)
        }

if __name__ == "__main__":
    # Test with random noise
    data = np.random.randn(19, 256) # 19 channels, 1 sec at 256Hz
    dmd = DMD_FeatureExtractor(rank=10)
    dmd.fit(data)
    feats = dmd.get_features()
    print("DMD Eigenvalues (Top 5):", feats['abs_eigenvalues'][:5])
