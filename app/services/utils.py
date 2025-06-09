import joblib
def save_scaler(scaler,path):
    joblib.dump(scaler,path)
def load_scaler(path):
    return joblib.load(path)
def extract_subject_id_from_filename(filename):
    if(filename.startswith("S") and filename[1:4].isdigit()):
        return int(filename[1:4])-1
    else:
        raise ValueError(f"Cannot extract subject_id from filename: {filename}")