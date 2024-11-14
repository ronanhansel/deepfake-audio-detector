import pandas as pd
from tqdm import tqdm
import os
import librosa
from scipy.signal import butter, lfilter
import random
import numpy as np
from spafe.features.lpc import lpc

def get_audio_duration(file_path):
    """
    Calculate the duration of an audio file.

    :param file_path: Path to the audio file (.wav)
    :return: Duration of the audio file (seconds)
    """
    try:
        duration = librosa.get_duration(filename=file_path)
        return duration
    except Exception as e:
        print(f"Error calculating duration for {file_path}: {e}")
        return np.nan

def remove_silence(y, sr, top_db=20):
    # Loại bỏ khoảng lặng ở đầu và cuối tín hiệu
    y_trimmed, _ = librosa.effects.trim(y, top_db=top_db)
    return y_trimmed

def bandpass_filter(y, sr, lowcut, highcut, order=3):
    nyq = 0.5 * sr  # Tần số Nyquist
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    y_filtered = lfilter(b, a, y)
    return y_filtered

def normalize_volume(y, desired_rms=0.1):
    # Tính RMS hiện tại của tín hiệu
    current_rms = np.sqrt(np.mean(y**2))
    # Tính tỷ lệ để điều chỉnh
    scalar = desired_rms / (current_rms + 1e-6)  # Thêm epsilon để tránh chia cho 0
    # Điều chỉnh tín hiệu
    y_normalized = y * scalar
    return y_normalized

def decrease_low_db(y, sr, threshold_db=-40, target_db=-80):
    """
    Minimise volumn below a threshold. This is useful to prevent environmental noise being considered by the model

    :param y: raw audio input (numpy array)
    :param sr: Sample rate (Hz)
    :param threshold_db: (e.g: -40 dB)
    :param target_db: the db to which all pieces below threshold will decrease (e.g: -80 dB)
    :return: (numpy array)
    """
    abs_y = np.abs(y)

    # Find the maximum ampltitude
    ref_amplitude = np.max(abs_y) if np.max(abs_y) > 0 else 1.0

    # calculate relative db
    y_db = 20 * np.log10(abs_y / ref_amplitude + 1e-10)

    mask = y_db < threshold_db

    desired_amplitude = 10 ** (target_db / 20) * ref_amplitude  # E.g: -80 dB
    y_adjusted = y.copy()

    y_adjusted[mask] = y_adjusted[mask] / (abs_y[mask] + 1e-10) * desired_amplitude

    return y_adjusted

def extract_features(y, sr, n_mfcc=13, n_chroma=3, n_spectral_contrast=5, lpc_order=12):
    """
    Trích xuất các đặc trưng từ tín hiệu âm thanh:
    - MFCC: 13 đặc trưng (trung bình theo thời gian)
    - Chroma: 3 đặc trưng (giảm từ 12 bin bằng cách nhóm và tính trung bình)
    - ZCR: 1 đặc trưng (trung bình)
    - Spectral Contrast: 6 đặc trưng (trung bình theo thời gian)
    - Spectral Centroid: 1 đặc trưng (trung bình)
    - Spectral Bandwidth: 1 đặc trưng (trung bình)
    - Spectral Rolloff: 1 đặc trưng (trung bình)
    - LPC: 12 đặc trưng (trung bình theo thời gian)

    Tổng cộng: 38 đặc trưng

    :param y: Tín hiệu âm thanh (numpy array, mono)
    :param sr: Tần số lấy mẫu (Hz)
    :param n_mfcc: Số lượng MFCCs (mặc định là 13)
    :param n_chroma: Số lượng nhóm Chroma cần lấy (mặc định là 3)
    :param n_spectral_contrast: Số lượng bands của Spectral Contrast (mặc định là 5 để có 6 đặc trưng)
    :param lpc_order: Độ bậc của LPC (mặc định là 12 để có 12 hệ số LPC)
    :return: Vector đặc trưng (numpy array) có độ dài 38
    """
    try:
        # 1. Trích xuất MFCCs và tính trung bình theo thời gian
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)  # Shape: (13, T)
        mfccs_mean = np.mean(mfccs, axis=1)  # (13,)
    except Exception as e:
        raise ValueError(f"Lỗi khi trích xuất MFCCs: {e}")
    if n_chroma > 0:
        try:
            # 2. Trích xuất Chroma Features, trung bình theo thời gian, nhóm lại
            chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_chroma=12)  # Shape: (12, T)
            chroma_mean_time = np.mean(chroma, axis=1)  # (12,)

            # Nhóm Chroma thành 3 nhóm, mỗi nhóm gồm 4 bin, tính trung bình mỗi nhóm
            if chroma_mean_time.shape[0] != 12:
                raise ValueError(f"Chroma bins không đúng kích thước. Mong đợi: 12, nhận được: {chroma_mean_time.shape[0]}")

            chroma_grouped = chroma_mean_time.reshape(3, 4).mean(axis=1)  # (3,)
        except Exception as e:
            raise ValueError(f"Lỗi khi trích xuất Chroma Features: {e}")
    # if zcr > 0:
    #     try:
    #         # 3. Trích xuất Zero Crossing Rate (ZCR) và tính trung bình
    #         zcr = librosa.feature.zero_crossing_rate(y)  # Shape: (1, T)
    #         zcr_mean = np.mean(zcr)  # Scalar
    #     except Exception as e:
    #         raise ValueError(f"Lỗi khi trích xuất ZCR: {e}")
    if n_spectral_contrast > 0:
        try:
            # 4. Trích xuất Spectral Contrast với 5 bands và tính trung bình theo thời gian
            spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_bands=n_spectral_contrast)  # Shape: (6, T)
            spectral_contrast_mean = np.mean(spectral_contrast, axis=1)  # (6,)
        except Exception as e:
            raise ValueError(f"Lỗi khi trích xuất Spectral Contrast: {e}")
    # try:
    #     # 5. Trích xuất Spectral Centroid và tính trung bình
    #     spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)  # Shape: (1, T)
    #     spectral_centroid_mean = np.mean(spectral_centroid)  # Scalar
    # except Exception as e:
    #     raise ValueError(f"Lỗi khi trích xuất Spectral Centroid: {e}")
    # try:
    #     # 6. Trích xuất Spectral Bandwidth và tính trung bình
    #     spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)  # Shape: (1, T)
    #     spectral_bandwidth_mean = np.mean(spectral_bandwidth)  # Scalar
    # except Exception as e:
    #     raise ValueError(f"Lỗi khi trích xuất Spectral Bandwidth: {e}")
    # try:
    #     # 7. Trích xuất Spectral Rolloff và tính trung bình
    #     spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)  # Shape: (1, T)
    #     spectral_rolloff_mean = np.mean(spectral_rolloff)  # Scalar
    # except Exception as e:
    #     raise ValueError(f"Lỗi khi trích xuất Spectral Rolloff: {e}")

    # try:
    #     # 8. Trích xuất LPC và tính trung bình theo thời gian sử dụng spafe
    #     lpc_features_tuple = lpc(y, fs=sr, order=lpc_order)  # Trả về tuple (lpc_coeffs, energies)

    #     # Giả sử rằng lpc_features_tuple[0] là mảng LPC coefficients
    #     lpc_coeffs = lpc_features_tuple[0]  # Shape: (num_frames, lpc_order) = (num_frames, 12)

    #     # Kiểm tra kiểu dữ liệu và hình dạng
    #     if not isinstance(lpc_coeffs, np.ndarray):
    #         raise ValueError("lpc_coeffs không phải là mảng NumPy.")
    #     if lpc_coeffs.ndim != 2:
    #         raise ValueError(f"Hệ số LPC không phải là mảng hai chiều. Nhận được: {lpc_coeffs.ndim} chiều")

    #     # Tính trung bình các hệ số LPC qua tất cả các khung thời gian
    #     lpc_mean = np.mean(lpc_coeffs, axis=0)  # (12,)

    #     # Kiểm tra độ dài của lpc_mean
    #     expected_lpc_length = lpc_order  # 12
    #     if lpc_mean.shape[0] != expected_lpc_length:
    #         raise ValueError(f"Số lượng hệ số LPC không khớp. Mong đợi: {expected_lpc_length}, nhận được: {lpc_mean.shape[0]}")
    # except Exception as e:
    #     raise ValueError(f"Lỗi khi trích xuất LPC: {e}")

    try:
        # 9. Kết hợp tất cả các đặc trưng
        features = np.concatenate((
            mfccs_mean,                     # 13
            # chroma_grouped,                 # 3
            # [zcr_mean],                     # 1
            # spectral_contrast_mean,         # 6
            # [spectral_centroid_mean],       # 1
            # [spectral_bandwidth_mean],      # 1
            # [spectral_rolloff_mean],        # 1
            # lpc_mean                        # 12
        ))  # Tổng cộng: 13 + 3 + 1 + 6 + 1 + 1 + 1 + 12 = 38

        # Kiểm tra tổng số đặc trưng
        if len(features) != 30:
            raise ValueError(f"Số lượng đặc trưng không khớp. Mong đợi: 30, nhận được: {len(features)}")
    except Exception as e:
        raise ValueError(f"Lỗi khi kết hợp các đặc trưng: {e}")

    return features

def process_audio_files(file_paths, labels, output_csv='features.csv', sample_size=100,
                        n_mfcc=13, n_chroma=3, n_spectral_contrast=5, lpc_order=12, resume=True):
    """
    Feature extraction from files and save to a CSV file.

    :param file_paths: List of paths to audio files
    :param labels: List of labels corresponding to the audio files
    :param output_csv: Path to the output CSV file
    :param sample_size: Number of audio files to randomly select for processing
    :param n_mfcc: Number of MFCCs (default is 13)
    :param n_chroma: Number of Chroma bands to extract (default is 3)
    :param n_spectral_contrast: Number of Spectral Contrast bands (default is 5 to get 6 features)
    :param lpc_order: Order of LPC (default is 12 to get 12 coefficients)
    :return: DataFrame containing the extracted features
    """
    # Check if the number of files is sufficient to select sample_size
    if sample_size is not None and len(file_paths) < sample_size:
        raise ValueError(f"Number of files is less than {sample_size}. Please check the file_paths list.")

    # Check the length of file_paths and labels
    if len(file_paths) != len(labels):
        raise ValueError("The number of file paths and labels must be the same.")

    # Ghép cặp file_paths và labels
    file_label_pairs = list(zip(file_paths, labels))

    # Chọn ngẫu nhiên sample_size cặp mà không lặp lại
    if sample_size is not None:
        selected_pairs = random.sample(file_label_pairs, sample_size)
    else:
        selected_pairs = file_label_pairs  # Xử lý toàn bộ nếu sample_size=None

    # Khởi tạo danh sách để lưu trữ các đặc trưng
    features_list = []

    # Định nghĩa các tên cột cho DataFrame với tên rõ ràng
    # Bao gồm tất cả các đặc trưng: MFCC, Chroma, ZCR, Spectral Contrast, Spectral Centroid, Spectral Bandwidth, Spectral Rolloff, LPC
    mfcc_columns = [f'mfcc_{i}_mean' for i in range(1, n_mfcc + 1)]  # mfcc_1_mean đến mfcc_13_mean
    # chroma_columns = [f'chroma_{i}_mean' for i in range(1, n_chroma + 1)]  # chroma_1_mean đến chroma_3_mean
    # zcr_column = ['zcr_mean']  # zcr_mean
    # spectral_contrast_columns = [f'spectral_contrast_{i}_mean' for i in range(1, n_spectral_contrast + 2)]  # spectral_contrast_1_mean đến spectral_contrast_6_mean
    # spectral_centroid_column = ['spectral_centroid_mean']  # spectral_centroid_mean
    # spectral_bandwidth_column = ['spectral_bandwidth_mean']  # spectral_bandwidth_mean
    # spectral_rolloff_column = ['spectral_rolloff_mean']  # spectral_rolloff_mean
    # lpc_columns = [f'lpc_{i}_mean' for i in range(1, lpc_order + 1)]  # lpc_1_mean đến lpc_12_mean

    # Tổng hợp tất cả các cột đặc trưng
    feature_columns = (
        mfcc_columns 
        # chroma_columns +
        # zcr_column +
        # spectral_contrast_columns +
        # spectral_centroid_column +
        # spectral_bandwidth_column +
        # spectral_rolloff_column +
        # lpc_columns
    )

    additional_columns = ['file_path', 'sampling_rate', 'label']
    df_columns = additional_columns + feature_columns
    # Check if the file exists
    if not os.path.exists(output_csv) or resume is False:
        # If the file does not exist, create it with the specified columns
        df_output = pd.DataFrame(columns=df_columns)
        df_output.to_csv(output_csv, index=False)
    else:
        # If the file exists, read from it
        df_output = pd.read_csv(output_csv)

    for idx, (file_path, label) in enumerate(tqdm(selected_pairs, desc="Processing audio files"), 1):
        if resume:
            # Check if the file has been processed before
            if df_output['file_path'].str.contains(file_path).any():
                print(f"File has been processed before: {file_path}")
                continue
        try:
            # Đọc tệp âm thanh dưới dạng mono và chuẩn hóa
            y, sr = librosa.load(file_path, sr=None, mono=True)

            # Tiền xử lý âm thanh
            y = remove_silence(y, sr)
            y = bandpass_filter(y, sr, lowcut=200.0, highcut=3400.0, order=4)
            y = normalize_volume(y, desired_rms=0.1)
            y = decrease_low_db(y, sr, threshold_db=-50, target_db=-80)

            # Trích xuất đặc trưng
            features = extract_features(y, sr, n_mfcc=n_mfcc, n_chroma=n_chroma,
                                        n_spectral_contrast=n_spectral_contrast,
                                        lpc_order=lpc_order)

            # Kiểm tra xem số lượng đặc trưng có đủ không
            expected_features_length = 30  # 13 + 3 + 1 + 6 + 1 + 1 + 1 + 12 = 38
            if len(features) != expected_features_length:
                raise ValueError(f"Số lượng đặc trưng không khớp. Mong đợi: {expected_features_length}, nhận được: {len(features)}")

            # Tạo một dictionary cho dòng dữ liệu hiện tại với tên cột rõ ràng
            data_dict = {
                'file_path': file_path,
                'sampling_rate': sr,
                'label': label
            }

            # Giải thích: Các đặc trưng được sắp xếp theo thứ tự:
            # 0-12: MFCC (13)
            # 13-15: Chroma (3)
            # 16: ZCR (1)
            # 17-22: Spectral Contrast (6)
            # 23: Spectral Centroid (1)
            # 24: Spectral Bandwidth (1)
            # 25: Spectral Rolloff (1)
            # 26-37: LPC (12)

            # Thêm MFCC
            for i in range(n_mfcc):
                data_dict[mfcc_columns[i]] = features[i]

            # # Thêm Chroma
            # for i in range(n_chroma):
            #     data_dict[chroma_columns[i]] = features[n_mfcc + i]

            # # Thêm ZCR
            # zcr_index = n_mfcc + n_chroma
            # data_dict[zcr_column[0]] = features[zcr_index]

            # # Thêm Spectral Contrast
            # spectral_contrast_start = zcr_index + 1
            # for i in range(n_spectral_contrast + 1):  # +1 vì n_spectral_contrast=5 tạo ra 6 đặc trưng
            #     data_dict[spectral_contrast_columns[i]] = features[spectral_contrast_start + i]

            # # Thêm Spectral Centroid
            # spectral_centroid_index = spectral_contrast_start + n_spectral_contrast + 1
            # data_dict[spectral_centroid_column[0]] = features[spectral_centroid_index]

            # # Thêm Spectral Bandwidth
            # spectral_bandwidth_index = spectral_centroid_index + 1
            # data_dict[spectral_bandwidth_column[0]] = features[spectral_bandwidth_index]

            # # Thêm Spectral Rolloff
            # spectral_rolloff_index = spectral_bandwidth_index + 1
            # data_dict[spectral_rolloff_column[0]] = features[spectral_rolloff_index]

            # # Thêm LPC
            # lpc_start = spectral_rolloff_index + 1
            # lpc_end = lpc_start + lpc_order  # lpc_order=12
            # lpc_mean = features[lpc_start:lpc_end]  # 12 hệ số LPC

            # if len(lpc_mean) != lpc_order:
            #     raise ValueError(f"Số lượng hệ số LPC không khớp. Mong đợi: {lpc_order}, nhận được: {len(lpc_mean)}")

            # for i in range(lpc_order):
            #     data_dict[lpc_columns[i]] = lpc_mean[i]

            # Thêm vào danh sách đặc trưng
            features_list.append(data_dict)

            # Save progress every 500 files
            if idx % 50 == 0:
                df_progress = pd.DataFrame(features_list)
                df_progress.to_csv(output_csv, mode='a', header=False, index=False)
                features_list.clear()
            if idx % 50 == 0 or idx == sample_size:
                print(f"Snapshot {idx}/{sample_size} files.")

        except Exception as e:
            print(f"Lỗi khi xử lý {file_path}: {e}")
            continue  # Bỏ qua tệp này và tiếp tục với tệp tiếp theo

    # Tạo DataFrame từ danh sách đặc trưng
    df_features = pd.DataFrame(features_list)

    # Ghi DataFrame vào CSV
    if resume:
        df_features.to_csv(output_csv, mode='a', header=False, index=False)
    else:
        df_features.to_csv(output_csv, mode='w', header=True, index=False)

    print(f"Saved features to: {output_csv}")

    return df_features

def create_sequences(features, sequence_length, overlap):
    """
    Creates sequences from feature data.

    Args:
        features: A NumPy array of shape (num_samples, num_features).
        sequence_length: The desired length of each sequence.
        overlap: The number of overlapping features between consecutive sequences.

    Returns:
        A tuple containing:
            - A NumPy array of shape (num_sequences, sequence_length, num_features).
            - A list of indices indicating the starting sample of each sequence.
    """
    sequences = []
    indices = []  # Store the starting index of each sequence
    for i in range(0, len(features) - sequence_length + 1, sequence_length - overlap):
        sequences.append(features[i: i + sequence_length])
        indices.append(i)  # Store the starting index
    return np.array(sequences), indices

if __name__ == '__main__':
    audio_paths = ['./content/LibriSeVoc/gt/3436_172171_000006_000008.wav', './content/LibriSeVoc/gt/4640_19187_000035_000006.wav', './content/LibriSeVoc/gt/3259_158083_000089_000000.wav', './content/LibriSeVoc/gt/7447_91186_000006_000003.wav', './content/LibriSeVoc/gt/3168_173564_000014_000004.wav', './content/LibriSeVoc/gt/412_126975_000023_000001.wav', './content/LibriSeVoc/gt/7059_77900_000031_000002.wav', './content/LibriSeVoc/gt/4267_287369_000038_000000.wav', './content/LibriSeVoc/gt/8226_274371_000024_000000.wav', './content/LibriSeVoc/gt/3240_131231_000084_000000.wav', './content/LibriSeVoc/gt/7800_283492_000037_000001.wav', './content/LibriSeVoc/gt/4680_16026_000032_000001.wav', './content/LibriSeVoc/gt/7278_91083_000003_000002.wav', './content/LibriSeVoc/gt/2092_145706_000029_000002.wav', './content/LibriSeVoc/gt/5867_48852_000039_000003.wav', './content/LibriSeVoc/gt/7517_100442_000008_000003.wav', './content/LibriSeVoc/gt/7178_34645_000006_000005.wav', './content/LibriSeVoc/gt/2836_5355_000038_000000.wav', './content/LibriSeVoc/gt/4088_158079_000153_000000.wav', './content/LibriSeVoc/gt/3664_178366_000004_000003.wav', './content/LibriSeVoc/gt/3699_47246_000004_000002.wav', './content/LibriSeVoc/gt/3879_174923_000008_000001.wav', './content/LibriSeVoc/gt/5867_48852_000008_000005.wav', './content/LibriSeVoc/gt/6078_54007_000065_000007.wav', './content/LibriSeVoc/gt/4195_186238_000025_000001.wav', './content/LibriSeVoc/gt/7800_283492_000037_000000.wav', './content/LibriSeVoc/gt/7800_283493_000019_000000.wav', './content/LibriSeVoc/gt/6880_216547_000054_000006.wav', './content/LibriSeVoc/gt/3168_173565_000022_000003.wav', './content/LibriSeVoc/gt/5750_100289_000025_000001.wav', './content/LibriSeVoc/gt/8063_274112_000196_000000.wav', './content/LibriSeVoc/gt/4014_186183_000027_000000.wav', './content/LibriSeVoc/gt/2843_152918_000003_000000.wav', './content/LibriSeVoc/gt/1355_39947_000024_000008.wav', './content/LibriSeVoc/gt/8770_295462_000038_000000.wav', './content/LibriSeVoc/gt/1841_179183_000020_000002.wav', './content/LibriSeVoc/gt/4088_158079_000157_000003.wav', './content/LibriSeVoc/gt/8609_283227_000012_000004.wav', './content/LibriSeVoc/gt/8324_286682_000025_000006.wav', './content/LibriSeVoc/gt/4406_16882_000024_000008.wav', './content/LibriSeVoc/gt/696_92939_000012_000001.wav', './content/LibriSeVoc/gt/1737_148989_000005_000001.wav', './content/LibriSeVoc/gt/8098_278252_000009_000000.wav', './content/LibriSeVoc/gt/6367_74004_000002_000002.wav', './content/LibriSeVoc/gt/5393_19218_000047_000001.wav', './content/LibriSeVoc/gt/3168_173564_000014_000005.wav', './content/LibriSeVoc/gt/3526_176653_000083_000004.wav', './content/LibriSeVoc/gt/5390_24512_000048_000001.wav', './content/LibriSeVoc/gt/2843_152918_000025_000003.wav', './content/LibriSeVoc/gt/412_126975_000080_000003.wav', './content/LibriSeVoc/gt/7402_90848_000041_000000.wav', './content/LibriSeVoc/gt/6454_93938_000032_000003.wav', './content/LibriSeVoc/gt/7078_271888_000040_000002.wav', './content/LibriSeVoc/gt/1926_147979_000006_000003.wav', './content/LibriSeVoc/gt/201_122255_000016_000000.wav', './content/LibriSeVoc/gt/1898_145715_000013_000000.wav', './content/LibriSeVoc/gt/32_4137_000023_000007.wav', './content/LibriSeVoc/gt/2092_145706_000010_000000.wav', './content/LibriSeVoc/gt/2416_152139_000059_000001.wav', './content/LibriSeVoc/gt/7367_86737_000121_000005.wav', './content/LibriSeVoc/gt/2514_149482_000023_000010.wav', './content/LibriSeVoc/gt/7078_271888_000006_000001.wav', './content/LibriSeVoc/gt/7367_86737_000132_000013.wav', './content/LibriSeVoc/gt/4088_158077_000106_000000.wav', './content/LibriSeVoc/gt/6880_216547_000054_000002.wav', './content/LibriSeVoc/gt/5867_48852_000035_000004.wav', './content/LibriSeVoc/gt/1363_139304_000022_000000.wav', './content/LibriSeVoc/gt/40_222_000033_000002.wav', './content/LibriSeVoc/gt/6454_107462_000024_000002.wav', './content/LibriSeVoc/gt/8108_280354_000009_000004.wav', './content/LibriSeVoc/gt/4680_16041_000017_000002.wav', './content/LibriSeVoc/gt/3235_28433_000017_000004.wav', './content/LibriSeVoc/gt/200_124139_000051_000001.wav', './content/LibriSeVoc/gt/39_121916_000002_000002.wav', './content/LibriSeVoc/gt/6415_100596_000028_000000.wav', './content/LibriSeVoc/gt/7367_86737_000117_000000.wav', './content/LibriSeVoc/gt/8088_284756_000073_000000.wav', './content/LibriSeVoc/gt/6078_54007_000041_000004.wav', './content/LibriSeVoc/gt/4195_186236_000004_000005.wav', './content/LibriSeVoc/gt/374_180298_000011_000003.wav', './content/LibriSeVoc/gt/4195_186237_000009_000000.wav', './content/LibriSeVoc/gt/1246_135815_000006_000002.wav', './content/LibriSeVoc/gt/40_121026_000201_000001.wav', './content/LibriSeVoc/gt/4830_25898_000006_000005.wav', './content/LibriSeVoc/gt/8838_298545_000021_000000.wav', './content/LibriSeVoc/gt/8088_284756_000080_000002.wav', './content/LibriSeVoc/gt/5778_54535_000002_000004.wav', './content/LibriSeVoc/gt/5867_48852_000005_000001.wav', './content/LibriSeVoc/gt/6064_300880_000032_000000.wav', './content/LibriSeVoc/gt/7302_86814_000058_000001.wav', './content/LibriSeVoc/gt/6181_216552_000017_000000.wav', './content/LibriSeVoc/gt/6078_54013_000043_000000.wav', './content/LibriSeVoc/gt/4051_11217_000008_000000.wav', './content/LibriSeVoc/gt/8465_246943_000019_000003.wav', './content/LibriSeVoc/gt/4267_287369_000010_000000.wav', './content/LibriSeVoc/gt/3830_12535_000015_000001.wav', './content/LibriSeVoc/gt/8123_275216_000004_000000.wav', './content/LibriSeVoc/gt/60_121082_000007_000005.wav', './content/LibriSeVoc/gt/5867_48852_000021_000001.wav', './content/LibriSeVoc/gt/7067_76048_000071_000001.wav', './content/LibriSeVoc/diffwave/8123_275216_000048_000005_gen.wav', './content/LibriSeVoc/diffwave/374_180298_000023_000001_gen.wav', './content/LibriSeVoc/diffwave/3374_298032_000004_000002_gen.wav', './content/LibriSeVoc/diffwave/8975_270782_000020_000005_gen.wav', './content/LibriSeVoc/diffwave/250_142276_000003_000009_gen.wav', './content/LibriSeVoc/diffwave/1069_133699_000067_000000_gen.wav', './content/LibriSeVoc/diffwave/7059_88364_000008_000001_gen.wav', './content/LibriSeVoc/diffwave/4680_16041_000024_000007_gen.wav', './content/LibriSeVoc/diffwave/200_124140_000019_000000_gen.wav', './content/LibriSeVoc/diffwave/2092_145706_000046_000001_gen.wav', './content/LibriSeVoc/diffwave/2092_145706_000030_000000_gen.wav', './content/LibriSeVoc/diffwave/1263_138246_000051_000000_gen.wav', './content/LibriSeVoc/diffwave/1069_133699_000066_000010_gen.wav', './content/LibriSeVoc/diffwave/7794_295955_000002_000018_gen.wav', './content/LibriSeVoc/diffwave/412_126975_000051_000004_gen.wav', './content/LibriSeVoc/diffwave/7505_258964_000009_000003_gen.wav', './content/LibriSeVoc/diffwave/254_145458_000013_000001_gen.wav', './content/LibriSeVoc/diffwave/4088_158077_000060_000000_gen.wav', './content/LibriSeVoc/diffwave/7511_102420_000009_000006_gen.wav', './content/LibriSeVoc/diffwave/1334_135589_000013_000003_gen.wav', './content/LibriSeVoc/diffwave/8580_287364_000048_000000_gen.wav', './content/LibriSeVoc/diffwave/1355_39947_000011_000002_gen.wav', './content/LibriSeVoc/diffwave/6454_120342_000017_000001_gen.wav', './content/LibriSeVoc/diffwave/2910_131096_000010_000000_gen.wav', './content/LibriSeVoc/diffwave/40_121026_000046_000000_gen.wav', './content/LibriSeVoc/diffwave/1098_133695_000013_000002_gen.wav', './content/LibriSeVoc/diffwave/7794_295948_000009_000001_gen.wav', './content/LibriSeVoc/diffwave/40_121026_000052_000001_gen.wav', './content/LibriSeVoc/diffwave/4214_7146_000051_000002_gen.wav', './content/LibriSeVoc/diffwave/78_369_000042_000003_gen.wav', './content/LibriSeVoc/diffwave/458_126305_000010_000003_gen.wav', './content/LibriSeVoc/diffwave/7505_258958_000033_000003_gen.wav', './content/LibriSeVoc/diffwave/8580_287364_000032_000003_gen.wav', './content/LibriSeVoc/diffwave/6385_34655_000018_000011_gen.wav', './content/LibriSeVoc/diffwave/8770_295462_000058_000000_gen.wav', './content/LibriSeVoc/diffwave/1363_139304_000019_000000_gen.wav', './content/LibriSeVoc/diffwave/6454_107462_000011_000000_gen.wav', './content/LibriSeVoc/diffwave/3235_28452_000011_000002_gen.wav', './content/LibriSeVoc/diffwave/5808_48608_000017_000004_gen.wav', './content/LibriSeVoc/diffwave/8226_274369_000007_000007_gen.wav', './content/LibriSeVoc/diffwave/8051_295385_000016_000001_gen.wav', './content/LibriSeVoc/diffwave/6880_216547_000058_000003_gen.wav', './content/LibriSeVoc/diffwave/3526_176653_000001_000006_gen.wav', './content/LibriSeVoc/diffwave/3879_174923_000003_000005_gen.wav', './content/LibriSeVoc/diffwave/7780_274562_000009_000001_gen.wav', './content/LibriSeVoc/diffwave/1263_141777_000010_000000_gen.wav', './content/LibriSeVoc/diffwave/5561_39621_000028_000000_gen.wav', './content/LibriSeVoc/diffwave/4051_11218_000003_000004_gen.wav', './content/LibriSeVoc/diffwave/8123_275216_000032_000001_gen.wav', './content/LibriSeVoc/diffwave/1116_137572_000024_000000_gen.wav', './content/LibriSeVoc/diffwave/3664_178366_000013_000000_gen.wav', './content/LibriSeVoc/diffwave/8838_298546_000011_000000_gen.wav', './content/LibriSeVoc/diffwave/200_126784_000068_000000_gen.wav', './content/LibriSeVoc/diffwave/7780_274562_000005_000006_gen.wav', './content/LibriSeVoc/diffwave/5322_7680_000006_000000_gen.wav', './content/LibriSeVoc/diffwave/7447_91187_000006_000003_gen.wav', './content/LibriSeVoc/diffwave/2910_131096_000033_000001_gen.wav', './content/LibriSeVoc/diffwave/2514_149482_000004_000002_gen.wav', './content/LibriSeVoc/diffwave/4406_16883_000013_000004_gen.wav', './content/LibriSeVoc/diffwave/6454_107462_000005_000000_gen.wav', './content/LibriSeVoc/diffwave/8975_270782_000023_000011_gen.wav', './content/LibriSeVoc/diffwave/7505_258964_000034_000001_gen.wav', './content/LibriSeVoc/diffwave/4406_16883_000006_000004_gen.wav', './content/LibriSeVoc/diffwave/7800_283478_000042_000000_gen.wav', './content/LibriSeVoc/diffwave/8838_298546_000021_000009_gen.wav', './content/LibriSeVoc/diffwave/8975_270782_000017_000004_gen.wav', './content/LibriSeVoc/diffwave/7447_91187_000014_000000_gen.wav', './content/LibriSeVoc/diffwave/8747_293952_000005_000002_gen.wav', './content/LibriSeVoc/diffwave/8051_119902_000026_000000_gen.wav', './content/LibriSeVoc/diffwave/7278_104730_000004_000000_gen.wav', './content/LibriSeVoc/diffwave/1183_128659_000012_000001_gen.wav', './content/LibriSeVoc/diffwave/446_123501_000016_000000_gen.wav', './content/LibriSeVoc/diffwave/1841_179183_000020_000001_gen.wav', './content/LibriSeVoc/diffwave/7367_86737_000123_000006_gen.wav', './content/LibriSeVoc/diffwave/6836_61803_000033_000000_gen.wav', './content/LibriSeVoc/diffwave/5339_14134_000069_000000_gen.wav', './content/LibriSeVoc/diffwave/7794_295948_000003_000003_gen.wav', './content/LibriSeVoc/diffwave/6880_216547_000053_000005_gen.wav', './content/LibriSeVoc/diffwave/2514_149482_000005_000006_gen.wav', './content/LibriSeVoc/diffwave/1502_122615_000036_000006_gen.wav', './content/LibriSeVoc/diffwave/1088_134315_000059_000001_gen.wav', './content/LibriSeVoc/diffwave/7278_246956_000005_000001_gen.wav', './content/LibriSeVoc/diffwave/4406_16882_000024_000000_gen.wav', './content/LibriSeVoc/diffwave/374_180299_000045_000002_gen.wav', './content/LibriSeVoc/diffwave/3526_176653_000002_000000_gen.wav', './content/LibriSeVoc/diffwave/125_121342_000091_000001_gen.wav', './content/LibriSeVoc/diffwave/3242_67153_000019_000000_gen.wav', './content/LibriSeVoc/diffwave/5867_48852_000086_000000_gen.wav', './content/LibriSeVoc/diffwave/8098_278278_000004_000003_gen.wav', './content/LibriSeVoc/diffwave/6064_56168_000002_000003_gen.wav', './content/LibriSeVoc/diffwave/4680_16026_000007_000001_gen.wav', './content/LibriSeVoc/diffwave/8609_262281_000014_000000_gen.wav', './content/LibriSeVoc/diffwave/5789_57158_000055_000002_gen.wav', './content/LibriSeVoc/diffwave/5322_7678_000006_000030_gen.wav', './content/LibriSeVoc/diffwave/412_126975_000056_000000_gen.wav', './content/LibriSeVoc/diffwave/1069_133709_000043_000033_gen.wav', './content/LibriSeVoc/diffwave/4406_16882_000008_000004_gen.wav', './content/LibriSeVoc/diffwave/8098_278252_000007_000000_gen.wav', './content/LibriSeVoc/diffwave/1841_150351_000002_000000_gen.wav', './content/LibriSeVoc/diffwave/374_180299_000029_000001_gen.wav']
    labels = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    features = process_audio_files(audio_paths, labels, n_mfcc=30, output_csv='test.csv')
    print(features)
