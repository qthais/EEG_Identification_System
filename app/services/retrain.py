
from pathlib import Path
import numpy as np
import joblib

from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler

from app.services.config import CHANNELS, DATA_DIR, MODEL_DIR, UPLOAD_DIR
from app.services.data_loader import load_eeg_split_by_time
from app.services.model import build_cnn2d_model
def retrainModel():
    X_train, y_train, X_val, y_val = load_eeg_split_by_time(UPLOAD_DIR, CHANNELS)
    n_classes = len(np.unique(np.concatenate([y_train, y_val])))
    # Standardization: Fit on train, transform on both train & test
    scaler = StandardScaler()
    # Reshape data for standardization (Flatten the frequency bins & time bins)
    num_train_samples, num_channels, freq_bins, time_bins = X_train.shape #(5096,5,33,9)
    print('f and t',freq_bins,time_bins)
    X_train_reshaped = X_train.reshape(-1, freq_bins * time_bins)  # Flatten spectrograms (25480,297)
    scaler.fit(X_train_reshaped)  # Fit only on train data
    joblib.dump(scaler, MODEL_DIR/'scaler.pkl') 
    # Apply Standardization to Train & Test Sets
    X_train = scaler.transform(X_train_reshaped).reshape(num_train_samples, num_channels, freq_bins, time_bins)
    X_val   = scaler.transform(X_val.reshape(-1, freq_bins * time_bins)).reshape(X_val.shape[0], num_channels, freq_bins, time_bins)

    # Add channel dimension for Conv2D (Convert to shape: (samples, height, width, channels))
    X_train = np.transpose(X_train, (0, 2, 3, 1))  # Shape: (samples, freq_bins, time_bins, num_channels)
    X_val   = np.transpose(X_val, (0, 2, 3, 1))
    #(5096,33,9,5)
    # Define input shape (num_channels, freq_bins, time_bins, 1)

    input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3])  # (freq_bins, time_bins, num_channels)
    model = build_cnn2d_model(input_shape, n_classes)
    model.summary()

    # Training Callbacks
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    checkpoint = ModelCheckpoint("app/models/best_cnn2d_model.keras", monitor='val_loss', save_best_only=True)
    lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1)

    # Train Model
    print("=== Training ===")
    model.fit(X_train, y_train,
                        epochs=200,
                        batch_size=64,
                        validation_data=(X_val, y_val),
                        callbacks=[early_stop, checkpoint, lr_scheduler])

    test_loss, test_acc = model.evaluate(X_val,y_val, verbose=2)
    print(f"=== Retraining complete. Final Test Accuracy: {test_acc} ===")

    return float(test_acc)
