import os
import numpy as np
import pandas as pd
import librosa
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import scipy
import scipy.signal

def extract_features_from_folders(base_dir, class_folders):
    features = []
    for label in class_folders:
        folder = os.path.join(base_dir, label)
        if not os.path.isdir(folder):
            print(f"Warning: Folder not found: {folder}")
            continue
        for file in os.listdir(folder):
            if file.lower().endswith('.wav'):
                file_path = os.path.join(folder, file)
                try:
                    y, sr = librosa.load(file_path, sr=None, duration=10.0)
                    duration = librosa.get_duration(y=y, sr=sr)
                    if duration < 1:
                        continue
                    # ZCR, RMS, Spectral features
                    zcr = np.mean(librosa.feature.zero_crossing_rate(y)[0])
                    rms = np.mean(librosa.feature.rms(y=y)[0])
                    centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)[0])
                    bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)[0])
                    rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)[0])
                    # MFCCs
                    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
                    mfcc_means = [np.mean(mfcc) for mfcc in mfccs]
                    # STFT features
                    stft = np.abs(librosa.stft(y))
                    stft_mean = np.mean(stft)
                    stft_std = np.std(stft)
                    stft_max = np.max(stft)
                    # CWT features (skip if pywt is not allowed)
                    cwt_mean = cwt_std = cwt_max = np.nan
                    feature_row = (
                        [file_path, zcr, rms, centroid, bandwidth, rolloff] +
                        mfcc_means +
                        [stft_mean, stft_std, stft_max, cwt_mean, cwt_std, cwt_max] +
                        [label]
                    )
                    features.append(feature_row)
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
    columns = (
        ['filepath', 'zcr', 'rms', 'centroid', 'bandwidth', 'rolloff'] +
        [f'mfcc_{i+1}' for i in range(13)] +
        ['stft_mean', 'stft_std', 'stft_max', 'cwt_mean', 'cwt_std', 'cwt_max'] +
        ['label']
    )
    return pd.DataFrame(features, columns=columns)

if __name__ == "__main__":
    base_dir = os.path.dirname(__file__)
    class_folders = ['artifact', 'normal']  # Only classify artifact vs normal
    df = extract_features_from_folders(base_dir, class_folders)
    if df.empty:
        print("No features extracted. Check your folder structure and .wav files.")
        exit(1)
    print(f"Extracted features shape: {df.shape}")

    # Output features to CSV for AI model training
    output_csv = os.path.join(base_dir, "features_artifact_vs_normal_with_stft_cwt.csv")
    df.to_csv(output_csv, index=False)
    print(f"Features saved to {output_csv}")

    # Train and evaluate a model to identify artifact
    X = df.drop(columns=['filepath', 'label'])
    y = (df['label'] == 'artifact').astype(int)  # 1 for artifact, 0 for normal

    if len(X) < 2 or len(y.unique()) < 2:
        print("Not enough data or classes to train/test a model.")
        exit(1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print("\nArtifact Detection Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['normal', 'artifact']))

    """
    Artifact Detection Classification Report:
              precision    recall  f1-score   support

      normal       1.00      0.86      0.92         7
    artifact       0.89      1.00      0.94         8

    accuracy                           0.93        15
   macro avg       0.94      0.93      0.93        15
weighted avg       0.94      0.93      0.93        15


    - precision: For each class, the proportion of predicted positives that are actually correct.

    - recall: For each class, the proportion of actual positives that are correctly identified.

    - f1-score: The harmonic mean of precision and recall. A balanced measure of model accuracy for each class.

    - support: The number of true samples of each class in the test set.

    - accuracy: Overall proportion of correct predictions.
    - macro avg: Average of the metrics for all classes, treating all classes equally.
    - weighted avg: Average of the metrics for all classes, weighted by the number of true instances for each class.

    """
