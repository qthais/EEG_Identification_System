import os
import mne
from app.services.config import CHANNELS, DATA_DIR, EDF_KEYWORD, SAMPLE_RATE, STRIDE, SUBJECT_PREFIX, TIME_WINDOW
from app.services.eeg_processing import eeg_to_spectrogram
import numpy as np

from app.services.utils import extract_subject_id_from_filename

def load_eeg_split_by_time(data_dir=DATA_DIR, channels=CHANNELS,
                           sample_rate=SAMPLE_RATE, time_window=TIME_WINDOW, stride=STRIDE):
    X_train, y_train = [], []
    X_val, y_val = [], []
    X_test, y_test = [], []

    edf_files = [f for f in os.listdir(data_dir) if f.endswith(".edf")]

    for edf_file in edf_files:
        edf_path = os.path.join(data_dir, edf_file)

        try:
            subject_id = extract_subject_id_from_filename(edf_file)
        except Exception as e:
            print(f"Skipping {edf_file}: {e}")
            continue

        try:
            raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
        except Exception as e:
            print(f"Error reading {edf_path}: {e}")
            continue

        raw.pick(channels)
        raw.filter(0.5, 40, fir_design='firwin', verbose=False)

        t0_events = [ann for ann in raw.annotations if ann['description'] == "T0"]
        if not t0_events:
            continue

        seg_length = int(time_window * sample_rate)
        stride_samples = int(stride * sample_rate)

        for ann in t0_events:
            start_sample = int(ann['onset'] * sample_rate)
            event_duration = int(ann['duration'] * sample_rate) if ann['duration'] > 0 else raw.n_times - start_sample

            segments = []
            for offset in range(0, event_duration - seg_length + 1, stride_samples):
                seg_start = start_sample + offset
                seg_end = seg_start + seg_length
                if seg_end > raw.n_times:
                    break
                segment = raw.get_data(start=seg_start, stop=seg_end).T
                spec = eeg_to_spectrogram(segment)
                segments.append(spec)

            # Split into train/val/test
            total = len(segments)
            train_end = int(0.6 * total)
            val_end = int(0.8 * total)

            for i, spec in enumerate(segments):
                if i < train_end:
                    X_train.append(spec)
                    y_train.append(subject_id)
                elif i < val_end:
                    X_val.append(spec)
                    y_val.append(subject_id)
                else:
                    X_test.append(spec)
                    y_test.append(subject_id)
    return (np.array(X_train), np.array(y_train),
            np.array(X_val), np.array(y_val),
            np.array(X_test), np.array(y_test))