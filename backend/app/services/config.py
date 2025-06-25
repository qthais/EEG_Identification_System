from pathlib import Path
LOGIN_DIR = "app/data/uploads"
SUBJECT_PREFIX = "S"
EDF_KEYWORD = "R01"
SAMPLE_RATE = 160  # EEG Sampling Rate
TIME_WINDOW = 3  # 3 seconds per segment
STRIDE = 0.3          # 1-second stride
CHANNELS = ['Oz..', 'Iz..','Cz..']  # 5 EEG Channels
N_CLASSES = 109

BASE_DIR = Path(__file__).resolve().parent.parent.parent
UPLOAD_DIR = BASE_DIR / "app" / "data" / "uploads"
MODEL_DIR = BASE_DIR / "app" / "models"
DATA_DIR=BASE_DIR / "app" / "data" / "raw"/"files"
