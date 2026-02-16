print("RUNNING FILE:", __file__)
import os
from pathlib import Path

import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm
from pydub import AudioSegment, effects


# -------------------------------
# Feature extraction
# -------------------------------
def extract_log_mel_spectrogram(
    y,
    sr,
    n_fft=2048,
    hop_length=512,
    n_mels=128
):
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        power=2.0
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)
    return log_mel.astype(np.float32)


# -------------------------------
# Process a single audio file
# -------------------------------
def process_file(
    path,
    out_dir,
    sr,
    duration,
    n_fft,
    hop_length,
    n_mels
):
    try:
        audio = AudioSegment.from_file(path)
    except Exception:
        return None

    # Convert to mono & target sample rate
    audio = audio.set_channels(1).set_frame_rate(sr)

    # Normalize loudness
    audio = effects.normalize(audio)

    # Convert to numpy float32 [-1, 1]
    samples = np.array(audio.get_array_of_samples())
    max_val = float(2 ** (8 * audio.sample_width - 1))
    y = samples.astype(np.float32) / max_val

    # Fixed-length trimming / padding
    desired_len = int(duration * sr)
    if len(y) < desired_len:
        y = np.pad(y, (0, desired_len - len(y)))
    elif len(y) > desired_len:
        start = (len(y) - desired_len) // 2
        y = y[start:start + desired_len]

    # Extract log-mel spectrogram
    feat = extract_log_mel_spectrogram(
        y,
        sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels
    )

    # Save feature
    label = Path(path).parent.name
    label_dir = Path(out_dir) / label
    label_dir.mkdir(parents=True, exist_ok=True)

    out_path = label_dir / (Path(path).stem + ".npy")
    np.save(out_path, feat)

    return str(out_path)


# -------------------------------
# Find audio files
# -------------------------------
def find_audio_files(data_dir):
    for root, _, files in os.walk(data_dir):
        for f in files:
            if f.lower().endswith(".wav"):
                yield os.path.join(root, f)


# -------------------------------
# Process entire dataset
# -------------------------------
def process_dataset(
    data_dir,
    out_dir,
    sr=22050,
    duration=3.0,
    n_fft=2048,
    hop_length=512,
    n_mels=128
):
    os.makedirs(out_dir, exist_ok=True)
    rows = []

    files = list(find_audio_files(data_dir))

    for f in tqdm(files, desc="Processing audio"):
        out_path = process_file(
            f,
            out_dir,
            sr,
            duration,
            n_fft,
            hop_length,
            n_mels
        )
        if out_path is None:
            continue

        label = Path(f).parent.name
        rows.append({
            "orig_path": f,
            "feature_path": out_path,
            "label": label
        })

    df = pd.DataFrame(rows)
    meta_path = Path(out_dir) / "metadata.csv"
    df.to_csv(meta_path, index=False)

    return meta_path


# -------------------------------
# Main
# -------------------------------
if __name__ == "__main__":
    DATASET_PATH = os.path.join(
        os.getcwd(),
        "IRMAS-TrainingData",
        "IRMAS-TrainingData"
    )

    PROCESSED_PATH = os.path.join(os.getcwd(), "processed")

    meta = process_dataset(
        data_dir=DATASET_PATH,
        out_dir=PROCESSED_PATH
    )

    print("Wrote metadata:", meta)
