import os
import glob
import numpy as np
import scipy.io as sio

class DEEDLoader:
    def __init__(self, data_dir, seq_len=3000):
        self.data_dir = data_dir
        self.seq_len = seq_len
        
        # Mapping derived from the Dream Emotion Evaluation Dataset (DEED) literature
        self.emotion_map = {
            'E0': 'I did not experience any dream or have no dream recall.',
            'E1': 'I had a negative affective dream experience.',
            'E2': 'I had a relatively negative affective dream experience.',
            'E3': 'I had a neutral affective dream experience.',
            'E4': 'I had a relatively positive affective dream experience.',
            'E5': 'I had a positive affective dream experience.'
        }
        
        try:
            from sentence_transformers import SentenceTransformer
            print("Loading NLP SentenceTransformer (all-MiniLM-L6-v2)...")
            self.nlp_model = SentenceTransformer('all-MiniLM-L6-v2')
        except ImportError:
            print("WARNING: sentence-transformers not installed. Text embeddings will be zeros.")
            self.nlp_model = None

    def load_data(self):
        mat_files = glob.glob(os.path.join(self.data_dir, "*.mat"))
        if not mat_files:
            raise FileNotFoundError(f"No .mat files found in {self.data_dir}")
        
        print(f"Found {len(mat_files)} DEED Multi-Modal samples.")
        
        X_data = []
        Y_text = []
        Y_emb = []
        
        for idx, f in enumerate(mat_files):
            basename = os.path.basename(f)
            parts = basename.split('_')
            
            emotion_code = parts[3] # E0, E1, etc.
            if emotion_code not in self.emotion_map:
                continue
                
            english_text = self.emotion_map[emotion_code]
            
            try:
                # Load the MATLAB array
                mat_contents = sio.loadmat(f)
                data = mat_contents['Data'] # Shape is typically (6, 186800)
                
                # Take the last `seq_len` samples to represent the end of the dream before waking
                if data.shape[1] > self.seq_len:
                    segment = data[:, -self.seq_len:]
                else:
                    pad_width = self.seq_len - data.shape[1]
                    segment = np.pad(data, ((0,0), (0, pad_width)))
                    
                # TimeGAN expects (seq_len, channels) => (3000, 6)
                segment = segment.T
                
                # Sanitize corrupted NaN values natively found in MAT files
                segment = np.nan_to_num(segment, nan=0.0)
                
                # NLP Encoding
                if self.nlp_model:
                    emb = self.nlp_model.encode(english_text)
                else:
                    emb = np.zeros(384, dtype=np.float32)
                    
                X_data.append(segment)
                Y_text.append(english_text)
                Y_emb.append(emb)
                
                if (idx + 1) % 50 == 0:
                    print(f"Parsed {idx + 1} / {len(mat_files)} brainwaves...")
                    
            except Exception as e:
                print(f"Failed to process {basename}: {e}")
                
        X_data = np.array(X_data, dtype=np.float32)
        Y_emb = np.array(Y_emb, dtype=np.float32)
        
        return X_data, Y_text, Y_emb

if __name__ == "__main__":
    loader = DEEDLoader(r"c:\Users\Sachin.R\Downloads\Dream GAN\dream_eeg\Dream EEG with emotion labels")
    X, Y_t, Y_e = loader.load_data()
    print("X Shape:", X.shape)
    print("Y_emb Shape:", Y_e.shape)
