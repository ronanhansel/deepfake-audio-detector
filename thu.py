import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras.models import load_model 
from utils import process_audio_files, create_sequences, AudioFeatureExtractor
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import classification_report
import pickle
import os


def get_audio_paths(folder_path, max_files=200, extensions=(".wav", ".mp3",".flac")):
    # Lấy danh sách file trong thư mục với các định dạng phù hợp
    audio_files = [
        os.path.join(folder_path, file)
        for file in os.listdir(folder_path)
        if file.lower().endswith(extensions)
    ]
    # Giới hạn số lượng file cần lấy
    return audio_files[:max_files]

# Ví dụ sử dụng
if __name__ == "__main__":
    real = []
    spoof = []
    folder_real = "E:/777/126732"  # Thay bằng đường dẫn thực tế
    folder_synthesized = "C:/Users/Trinh Ha Phuong/OneDrive - Hanoi University of Science and Technology/Documents/intro to AI/deepfake-audio-detector/voice dataset/LibriSeVoc/wavegrad"
    audio_paths_1 = get_audio_paths(folder_real)
    audio_paths_0 = get_audio_paths(folder_synthesized)

    print(f"Found {len(audio_paths_1)} audio files.")
    print(f"Found {len(audio_paths_0)} audio files.")
    for path in audio_paths_1:
        if AudioFeatureExtractor.get_audio_duration(path) <= 10:
            real.append(path)
    
    for path in audio_paths_0:
        if AudioFeatureExtractor.get_audio_duration(path) <= 10:
            spoof.append(path)
    audio_paths = real + spoof
    labels = [1] * len(real) + [0] * len(spoof)

    print(00)
    #process_audio_files(file_paths= audio_paths, n_mfcc=30, labels=labels, output_csv='./output/selected_files.csv', resume=False)
    print(10)

# Load the model
model = load_model('best_model.keras')
model.summary()

def evaluate_model(model, audio_paths, labels):
    # Load and prepare test data
    
    features = process_audio_files(file_paths= audio_paths, n_mfcc=30, labels=labels, output_csv='./output/test1.csv', resume=False)

    print(11)
    if features is None or len(features) == 0:
        print("No data returned from process_audio_files.")

    
    df_test = pd.read_csv('./output/test1.csv')

    # Extract features and labels
    features = df_test.drop(columns=['label', 'audio_path'])
    labels = df_test['label']
    print('Test data:', features.shape, labels.shape)

    # 1. Create sequences (same as training)
    sequence_length = 10
    overlap = 5
    sequences, indices = create_sequences(features, sequence_length, overlap)
    labels = labels[indices]

    # 2. Pad sequences (make sure maxlen matches training)
    padded_sequences = pad_sequences(
        sequences, 
        maxlen=2,  # Changed from 2 to match training
        padding="pre", 
        truncating="post"
    )

    # 3. Load and apply scaler
    with open('./output/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)

    # Reshape and scale (exactly as in training)
    num_samples, seq_len, num_features = padded_sequences.shape
    padded_sequences_reshaped = padded_sequences.reshape(num_samples, -1)
    padded_sequences_scaled = scaler.transform(padded_sequences_reshaped)
    padded_sequences = padded_sequences_scaled.reshape(num_samples, seq_len, num_features)

    # 4. Evaluate
    y_pred = model.predict(padded_sequences)
    y_pred_binary = (y_pred > 0.5).astype(int)
    
    # Print metrics
    loss, accuracy = model.evaluate(padded_sequences, labels)
    print(f"\nModel loss: {loss:.4f}")
    print(f"Model accuracy: {accuracy * 100:.2f}%")
    
    # Additional metrics
    print("\nClassification Report:")
    print(classification_report(labels, y_pred_binary))
    
    return y_pred, labels
# Usage

# Create a list of audio paths and corresponding labels


evaluate_model(model, audio_paths=audio_paths, labels=labels)
#dự đoán của model