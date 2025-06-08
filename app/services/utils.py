import joblib
def save_scaler(scaler,path):
    joblib.dump(scaler,path)
def load_scaler(path):
    return joblib.load(path)