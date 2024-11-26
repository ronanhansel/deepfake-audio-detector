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

#lấy path của audio files
import os

def get_audio_paths(folder_path, max_files=500, extensions=(".wav", ".mp3",".flac")):
    """
    Lấy đường dẫn của các file audio trong thư mục.
    
    Args:
        folder_path (str): Đường dẫn tới thư mục chứa các file audio.
        max_files (int): Số lượng file tối đa cần lấy.
        extensions (tuple): Các phần mở rộng hợp lệ của file audio.
        
    Returns:
        list: Danh sách đường dẫn của các file audio.
    """
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
    

# Load the model
model = load_model('best_model.keras')
model.summary()
#features = process_audio_files(audio_paths, n_mfcc=30, labels=labels, output_csv='./output/test.csv', resume=False)

# Create a DataFrame from selected_gt_files and selected_diffwave_files
selected_files_df = pd.DataFrame({
    'file_path': audio_paths_1 + audio_paths_0,
    'label': [1] * len(audio_paths_1) + [0] * len(audio_paths_0)
})

selected_files_df.to_csv('./output/selected_files.csv', index=False)

# Create a list of audio paths and corresponding labels
audio_paths = audio_paths_1 + audio_paths_0
labels = [1] * len(audio_paths_1) + [0] * len(audio_paths_0)



def extract_mfcc_features(audio_path, sr, n_mfcc):
    """
    Trích xuất đặc trưng MFCC từ file audio.
    
    Args:
        audio_path (str): Đường dẫn tới file audio.
        sr (int): Tần số lấy mẫu (sampling rate).
        n_mfcc (int): Số MFCC features cần trích xuất.
        
    Returns:
        list: Trung bình các giá trị của MFCC (1 giá trị cho mỗi MFCC).
    """
    try:
        # Load audio
        audio, _ = librosa.load(audio_path, sr=sr)
        # Trích xuất MFCC
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
        # Tính trung bình từng MFCC
        mfcc_mean = mfcc.mean(axis=1)
        return mfcc_mean.tolist()
    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        return [None] * n_mfcc

def create_mfcc_csv(folder_path, output_csv):
    """
    Tạo file CSV chứa thông số MFCC của các file audio trong thư mục.
    
    Args:
        folder_path (str): Đường dẫn tới thư mục chứa các file audio.
        output_csv (str): Đường dẫn để lưu file CSV.
        max_files (int): Số lượng file tối đa để xử lý.
    """
    # Lấy danh sách file audio
    #audio_paths = get_audio_paths(folder_path, max_files=max_files)
    
    # Danh sách lưu trữ dữ liệu
    data = []
    
    # Duyệt qua từng file audio và trích xuất MFCC
    for audio_path,label in list(zip(audio_paths,labels)):
        mfcc_features = extract_mfcc_features(audio_path,sr=24000,n_mfcc= 30)
        if mfcc_features[0] is not None:  # Kiểm tra nếu không có lỗi
            data.append([audio_path] + mfcc_features + [label])  
    
    # Tạo DataFrame từ dữ liệu
    columns = ["audio_path"] + [f"mfcc_{i+1}" for i in range(len(data[0]) - 2)] + ["label"]
    df = pd.DataFrame(data, columns=columns)
    
    # Lưu DataFrame thành CSV
    df.to_csv(output_csv, index=False)
    print(f"CSV file created at: {output_csv}")




def evaluate_model(model, audio_paths, labels):
    # Load and prepare test data
    
    #features = process_audio_files(audio_paths, n_mfcc=30, labels=labels, output_csv='./output/test.csv', resume=False)

    # print(1)
    # if features is None or len(features) == 0:
    #     print("No data returned from process_audio_files.")

    create_mfcc_csv(folder_path = audio_paths, output_csv = './output/test.csv')
    df_test = pd.read_csv('./output/test.csv')

    # Extract features and labels
    features = df_test.drop(columns=[ 'audio_path'])
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