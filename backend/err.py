import os
import numpy as np
import joblib
import tensorflow as tf
from keras.models import Model, load_model
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt

DATA_DIR = "backend/app/data/raw/files/"
SUBJECT_PREFIX = "S"
EDF_KEYWORD = "R01"
SAMPLE_RATE = 160  # EEG Sampling Rate
TIME_WINDOW = 3  # 3 seconds per segment
STRIDE = 0.3          # 1-second stride
CHANNELS = ['Iz..','O2..','Oz..']  # 5 EEG Channels
N_CLASSES = 109
# ----------------- Hàm tính EER -----------------
def compute_eer(y_true, y_scores):
    """
    Tính Equal Error Rate (EER).
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    fnr = 1 - tpr
    idx = np.nanargmin(np.abs(fnr - fpr))
    eer = fpr[idx]
    thr = thresholds[idx]
    return eer, thr, fpr, tpr

# ----------------- Cosine similarity -----------------
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)

# ----------------- Hàm evaluate EER -----------------
def evaluate_eer(model_path, scaler_path, X_test, y_test, plot_path="backend/app/models/eer_roc.png"):
    """
    Tính EER cho toàn bộ subject.
    - model_path: đường dẫn model .keras đã train (softmax)
    - scaler_path: scaler.pkl (đã fit trên train)
    - X_test, y_test: dữ liệu test
    """
    # Load model & scaler
    model = load_model(model_path)
    scaler = joblib.load(scaler_path)

    # Chuẩn hóa lại dữ liệu test
    num_samples, num_channels, freq_bins, time_bins = X_test.shape
    X_test_flat = X_test.reshape(-1, freq_bins * time_bins)
    X_test = scaler.transform(X_test_flat).reshape(num_samples, num_channels, freq_bins, time_bins)
    X_test = np.transpose(X_test, (0, 2, 3, 1))  # (samples, freq, time, channels)

    # Trích embedding từ layer trước softmax
    feature_model = Model(inputs=model.input, outputs=model.layers[-2].output)
    embeddings = feature_model.predict(X_test, verbose=0)

    # Tạo pair similarity
    genuine_scores, impostor_scores = [], []
    for i in range(len(y_test)):
        for j in range(i+1, len(y_test)):
            score = cosine_similarity(embeddings[i], embeddings[j])
            if y_test[i] == y_test[j]:
                genuine_scores.append(score)
            else:
                impostor_scores.append(score)

    # Tính EER
    y_true = np.array([1]*len(genuine_scores) + [0]*len(impostor_scores))
    y_scores = np.array(genuine_scores + impostor_scores)

    eer, thr, fpr, tpr = compute_eer(y_true, y_scores)

    print(f"Equal Error Rate (EER): {eer:.6f} at threshold {thr:.6f}")

    # Vẽ ROC
    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC (EER = {eer:.4f})")
    plt.plot([0,1],[0,1],"k--")
    plt.xlabel("False Acceptance Rate (FAR)")
    plt.ylabel("True Acceptance Rate (1 - FRR)")
    plt.title("Authentication ROC Curve (All Subjects)")
    plt.legend()
    plt.savefig(plot_path, dpi=200)
    plt.show()

    return eer, thr

# ----------------- MAIN -----------------
if __name__ == "__main__":
    # Đường dẫn file của bạn
    MODEL_PATH = "backend/app/models/best_cnn2d_model.keras"
    SCALER_PATH = "backend/app/models/scaler.pkl"
    DATA_DIR = "backend/app/data/raw/files/"

    # Import lại hàm load dữ liệu từ file train
    from updateCnn2d import load_eeg_split_by_time

    # Load dữ liệu (chia sẵn train/val/test)
    _, _, _, _, X_test, y_test = load_eeg_split_by_time(
        DATA_DIR, SUBJECT_PREFIX, EDF_KEYWORD, CHANNELS
    )

    evaluate_eer(MODEL_PATH, SCALER_PATH, X_test, y_test)
