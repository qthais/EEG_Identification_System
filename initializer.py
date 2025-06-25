import os
import mne

root_dir = "app/data/raw/files"   


for subj in sorted(os.listdir(root_dir)):
    subj_dir = os.path.join(root_dir, subj)
    if not os.path.isdir(subj_dir):
        continue

    # pick the first *R01.edf (ignore .event)
    edfs = [f for f in os.listdir(subj_dir)
            if f.upper().endswith("R01.EDF") and not f.upper().endswith(".EVENT")]
    if not edfs:
        continue

    raw = mne.io.read_raw_edf(
        os.path.join(subj_dir, edfs[0]),
        preload=True, verbose=False
    )

    # Crop segments
    endTime=raw.times[-1]
    seg48 = raw.copy().crop(tmin=0.0,  tmax=48.0, include_tmax=False)
    seg12 = raw.copy().crop(tmin=48.0, tmax=endTime, include_tmax=False)

    # Export using raw.export (fmt='edf')
    out48 = os.path.join(subj_dir, f"{subj}_48s.edf")
    out12 = os.path.join(subj_dir, f"{subj}_12s.edf")

    seg48.export(out48, fmt='edf', overwrite=True)
    seg12.export(out12, fmt='edf', overwrite=True)

    print(f"[OK] {subj}: exported {os.path.basename(out48)} & {os.path.basename(out12)}")
