import numpy as np
import pandas as pd
import librosa.display
import matplotlib.pyplot as plt

# Load metadata
df = pd.read_csv("processed/metadata.csv")

# Pick ONE spectrogram file
spec_path = df.iloc[0]["feature_path"]   # first sample
print("Loading:", spec_path)

# Load spectrogram (.npy)
spec = np.load(spec_path)

# Plot
plt.figure(figsize=(8, 4))
librosa.display.specshow(
    spec,
    x_axis="time",
    y_axis="mel",
    cmap="magma"
)
plt.colorbar(format="%+2.0f dB")
plt.title("Log-Mel Spectrogram")
plt.tight_layout()
plt.show()
print(spec.shape)
print(spec.min(), spec.max())
