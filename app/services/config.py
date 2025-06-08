DATA_DIR = "files/"
SUBJECT_PREFIX = "S"
EDF_KEYWORD = "R01"
SAMPLE_RATE = 160  # EEG Sampling Rate
TIME_WINDOW = 3  # 3 seconds per segment
STRIDE = 0.3          # 1-second stride
CHANNELS = ['Oz..', 'Iz..','P7..','Cz..']  # 5 EEG Channels
N_CLASSES = 109