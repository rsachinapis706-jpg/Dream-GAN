import os
import glob
import numpy as np
import mne
import docx

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

class DondersLoader:
    def __init__(self, donders_root: str, seq_len: int = 3000):
        self.donders_root = os.path.abspath(donders_root)
        self.psg_dir = os.path.join(self.donders_root, 'Data', 'PSG')
        self.reports_dir = os.path.join(self.donders_root, 'Data', 'Reports')
        self.seq_len = seq_len
        
        # We will use a fast SentenceTransformer model to convert English paragraphs to 384-D math vectors
        if SentenceTransformer is not None:
            print("Loading NLP Model (SentenceTransformer)...")
            self.nlp_model = SentenceTransformer('all-MiniLM-L6-v2')
        else:
            print("SentenceTransformer not installed, embeddings will be mocked.")
            self.nlp_model = None

    def _extract_text_from_docx(self, docx_path: str) -> str:
        try:
            doc = docx.Document(docx_path)
            full_text = []
            for para in doc.paragraphs:
                text = para.text.strip()
                if text:
                    full_text.append(text)
            return " ".join(full_text)
        except Exception as e:
            print(f"Error reading {docx_path}: {e}")
            return ""

    def load_multimodal_data(self):
        """
        Loads paired EEG/ECG and Semantic Text Embeddings.
        Returns:
            X (np.ndarray): Shape (batch, seq_len, 7) [6 EEG + 1 ECG channel]
            Y_text (list): Raw English dream reports
            Y_emb (np.ndarray): Shape (batch, 384) [Semantic math vectors]
        """
        # Find all EDF files
        edf_files = glob.glob(os.path.join(self.psg_dir, '**', '*.edf'), recursive=True)
        print(f"Found {len(edf_files)} EDF recordings in Donders Database.")
        
        X_data = []
        Y_text = []
        Y_emb = []
        
        # The channels we care about: Frontal, Central, Occipital, and Heart Rate
        target_channels = ['F3', 'F4', 'C3', 'C4', 'O1', 'O2', 'ECG']
        
        for edf_path in edf_files:
            # Figure out the subject and recording ID to match the docx
            # e.g .../s_04/c_05/morningnap_singlepart.edf
            parts = edf_path.split(os.sep)
            s_id = parts[-3]
            c_id = parts[-2]
            
            # Find matching docx files in Reports folder
            docx_pattern = os.path.join(self.reports_dir, s_id, c_id, '*.docx')
            docx_files = glob.glob(docx_pattern)
            
            if not docx_files:
                print(f"No textual dream report found for {s_id}/{c_id}. Skipping.")
                continue
                
            # Read and combine all text from matching docx files
            combined_text = ""
            for doc_file in docx_files:
                combined_text += self._extract_text_from_docx(doc_file) + " "
                
            if not combined_text.strip():
                continue
                
            print(f"\nProcessing {s_id}/{c_id}...")
            print(f"Dream Report snippet: '{combined_text[:100]}...'")
            
            # Load the biological data
            try:
                raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False, encoding='latin1')
                # Ensure all target channels exist
                channels_present = [ch for ch in target_channels if ch in raw.ch_names]
                if len(channels_present) != len(target_channels):
                    missing = set(target_channels) - set(channels_present)
                    print(f"Missing channels {list(missing)}. Filling them with zeros.")
                    
                # We want 100Hz standard. Resample if necessary.
                if raw.info['sfreq'] != 100:
                    raw.resample(100)
                    
                raw.pick_channels(channels_present)
                data = raw.get_data() # shape: (channels, total_time)
                
                # Reorder and pad to ensure exact 7 channels
                ordered_data = np.zeros((len(target_channels), data.shape[1]), dtype=np.float32)
                for i, ch in enumerate(target_channels):
                    if ch in channels_present:
                        idx = channels_present.index(ch)
                        ordered_data[i, :] = data[idx, :]
                
                # Create NLP vector once per patient
                if self.nlp_model:
                    emb = self.nlp_model.encode(combined_text.strip())
                else:
                    emb = np.zeros(384, dtype=np.float32)

                # DATA AUGMENTATION: Extract the last 5 minutes (300 seconds = 30000 samples @ 100Hz)
                max_samples = 30000
                if ordered_data.shape[1] > max_samples:
                    dream_segment = ordered_data[:, -max_samples:]
                else:
                    dream_segment = ordered_data
                
                # Rolling window: seq_len = 3000 (30s), step = 200 (2s overlap)
                step = 200
                if dream_segment.shape[1] < self.seq_len:
                    pad_width = self.seq_len - dream_segment.shape[1]
                    dream_segment = np.pad(dream_segment, ((0,0), (0, pad_width)))
                    
                windows_extracted = 0
                for start in range(0, dream_segment.shape[1] - self.seq_len + 1, step):
                    segment = dream_segment[:, start:start + self.seq_len].T
                    X_data.append(segment)
                    Y_text.append(combined_text.strip())
                    Y_emb.append(emb)
                    windows_extracted += 1
                
                print(f"Extracted {windows_extracted} overlapping time-windows (Data Augmentation).")
                
            except Exception as e:
                print(f"Failed to process EEG {edf_path}: {e}")
                
        X_data = np.array(X_data, dtype=np.float32)
        Y_emb = np.array(Y_emb, dtype=np.float32)
        
        print("\n--- DONDERS MULTIMODAL EXTRACTION COMPLETE ---")
        print(f"Total Paired Samples: {len(X_data)}")
        if len(X_data) > 0:
            print(f"Physiology Shape (X):   {X_data.shape} -> (Batch, TimeSteps, Channels)")
            print(f"Semantic Shape (Y_emb): {Y_emb.shape} -> (Batch, NLP_Vector_Size)")
            
        return X_data, Y_text, Y_emb

if __name__ == "__main__":
    loader = DondersLoader(donders_root=r"C:\Users\Sachin.R\Downloads\Dream GAN\Dream_Database_Donders\Extracted")
    X, texts, embs = loader.load_multimodal_data()
