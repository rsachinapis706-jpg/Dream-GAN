import mne

paths = [
    r"Dream_Database_Donders\Extracted\Data\PSG\s_00\c_00\morningnap_singlepart.edf",
    r"Dream_Database_Donders\Extracted\Data\PSG\s_01\c_01\morningnap_singlepart.edf"
]

for p in paths:
    print(f"\n--- Checking {p} ---")
    try:
        raw = mne.io.read_raw_edf(p, preload=False, verbose=False)
        print("Channels:")
        print(raw.ch_names)
    except Exception as e:
        print(e)
