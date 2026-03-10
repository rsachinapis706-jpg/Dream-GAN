import os
import mne

# Suppress MNE warnings for cleaner output
mne.set_log_level('WARNING')

def inspect_sleep_edf(data_dir):
    print(f"Inspecting Sleep-EDF Directory: {data_dir}\n")
    
    rec_files = [f for f in os.listdir(data_dir) if f.endswith('.rec')]
    rec_files.sort()
    
    if not rec_files:
        print("No .rec files found!")
        return

    # Inspect the first file
    sample_file = os.path.join(data_dir, rec_files[0])
    hyp_file = sample_file.replace('.rec', '.hyp')
    
    print(f"Loading sample record: {rec_files[0]}")
    edf_file = sample_file.replace('.rec', '.edf')
    try:
        # MNE strictly checks the extension, so we rename temporarily
        os.rename(sample_file, edf_file)
        raw = mne.io.read_raw_edf(edf_file, preload=False)
        print("\n--- Physical Channels & Sampling Rates ---")
        print(f"Total Channels: {len(raw.ch_names)}")
        print(f"Channel Names: {raw.ch_names}")
        print(f"Sampling Rate: {raw.info['sfreq']} Hz")
        print(f"Duration: {raw.times[-1] / 3600:.2f} hours")
        
        print("\n--- Checking Annotations (Hypnogram) ---")
        if os.path.exists(hyp_file):
            print(f"Found corresponding hypnogram: {os.path.basename(hyp_file)}")
            try:
                import pyedflib
                f_hyp = pyedflib.EdfReader(hyp_file)
                n_channels = f_hyp.signals_in_file
                print(f"Hypnogram File EDF Channels: {n_channels}")
                
                for i in range(n_channels):
                    ch_name = f_hyp.getLabel(i)
                    sfreq = f_hyp.getSampleFrequency(i)
                    data = f_hyp.readSignal(i)
                    print(f"  Channel {i}: {ch_name} | fs={sfreq}Hz | Len: {len(data)}")
                    print(f"  Sample Data (first 30): {data[:30]}")
                    unique_vals = list(set(data))
                    print(f"  Unique Values: {unique_vals}")
                f_hyp._close()
            except Exception as e:
                print(f"Failed to reverse engineer hypnogram: {e}")
        else:
            print("No matching .hyp file found.")
            
    except Exception as e:
        print(f"Failed to load EDF file. Error: {e}")
    finally:
        if os.path.exists(edf_file):
            os.rename(edf_file, sample_file)

if __name__ == "__main__":
    target_dir = r"c:\Users\Sachin.R\Downloads\Dream GAN\sleep-edf-database-1.0.0\sleep-edf-database-1.0.0"
    inspect_sleep_edf(target_dir)
