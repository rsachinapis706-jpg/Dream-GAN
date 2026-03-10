import mne
edf_path = r"Dream_Database_Donders\Extracted\Data\PSG\s_04\c_05\morningnap_singlepart.edf"
raw = mne.io.read_raw_edf(edf_path, preload=False, verbose=False)
ch_names = raw.ch_names
print("All Channels:")
for ch in ch_names:
    print(" -", ch)

ecg_channels = [ch for ch in ch_names if 'ECG' in ch.upper()]
print("\nECG Channels found:", ecg_channels)
