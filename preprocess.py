

import os
import librosa
import numpy as np

# =========================
# GLOBAL CONFIGURATION
# =========================

SAMPLE_RATE = 22050        # Standard for audio ML
DURATION = 3.0             # seconds (fixed length)
SAMPLES_PER_TRACK = int(SAMPLE_RATE * DURATION)

N_MELS = 128               # Mel bands
N_FFT = 2048               # FFT window size
HOP_LENGTH = 512           # Hop length

IMG_HEIGHT = 128
IMG_WIDTH = 128


# =========================
# AUDIO LOADING
# =========================

def load_audio(audio_path):
    """
    Load audio file safely.
    Converts stereo to mono.
    """

    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    signal, sr = librosa.load(
        audio_path,
        sr=SAMPLE_RATE,
        mono=True
    )

    return signal, sr


# =========================
# FIX AUDIO LENGTH
# =========================

def fix_audio_length(signal):
    """
    Ensures all audio signals have equal length.
    - Trim if longer
    - Pad with zeros if shorter
    """

    if len(signal) > SAMPLES_PER_TRACK:
        signal = signal[:SAMPLES_PER_TRACK]

    elif len(signal) < SAMPLES_PER_TRACK:
        padding = SAMPLES_PER_TRACK - len(signal)
        signal = np.pad(signal, (0, padding), mode="constant")

    return signal


# =========================
# MEL SPECTROGRAM
# =========================

def compute_mel_spectrogram(signal):
    """
    Convert audio signal to Mel-Spectrogram
    """

    mel_spec = librosa.feature.melspectrogram(
        y=signal,
        sr=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS
    )

    mel_db = librosa.power_to_db(mel_spec, ref=np.max)

    return mel_db


# =========================
# RESIZE SPECTROGRAM
# =========================

def resize_spectrogram(spec):
    """
    Resize spectrogram to fixed 128x128
    """

    spec = spec[:IMG_HEIGHT, :IMG_WIDTH]

    if spec.shape[1] < IMG_WIDTH:
        pad_width = IMG_WIDTH - spec.shape[1]
        spec = np.pad(
            spec,
            ((0, 0), (0, pad_width)),
            mode="constant"
        )

    return spec


# =========================
# NORMALIZATION
# =========================

def normalize_spectrogram(spec):
    """
    Normalize spectrogram to range [0, 1]
    """

    spec = spec - spec.min()
    if spec.max() != 0:
        spec = spec / spec.max()

    return spec


# =========================
# MAIN FUNCTION (USED BY TRAINING)
# =========================

def extract_mel(audio_path):
    """
    COMPLETE PREPROCESSING PIPELINE

    Steps:
    1. Load audio
    2. Fix length
    3. Compute Mel-Spectrogram
    4. Resize
    5. Normalize

    Returns:
    --------
    mel_spec : np.ndarray (128 x 128)
    """

    # 1. Load
    signal, sr = load_audio(audio_path)

    # 2. Fix length
    signal = fix_audio_length(signal)

    # 3. Mel-Spectrogram
    mel_spec = compute_mel_spectrogram(signal)

    # 4. Resize
    mel_spec = resize_spectrogram(mel_spec)

    # 5. Normalize
    mel_spec = normalize_spectrogram(mel_spec)

    return mel_spec


# =========================
# OPTIONAL: MFCC FEATURES
# (Not used for CNN, but useful)
# =========================

def extract_mfcc(audio_path, n_mfcc=20):
    """
    Extract MFCC features (optional)
    """

    signal, sr = load_audio(audio_path)
    signal = fix_audio_length(signal)

    mfcc = librosa.feature.mfcc(
        y=signal,
        sr=sr,
        n_mfcc=n_mfcc
    )

    mfcc = mfcc.T
    return mfcc


# =========================
# DEBUG TEST (OPTIONAL)
# =========================

if __name__ == "__main__":
    # Simple test to verify preprocessing
    test_audio = "dataset/audio/sample.wav"

    if os.path.exists(test_audio):
        mel = extract_mel(test_audio)
        print("Preprocessing OK")
        print("Shape:", mel.shape)
    else:
        print("Test audio not found")
