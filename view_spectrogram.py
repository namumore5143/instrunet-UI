import numpy as np
import matplotlib.pyplot as plt
import os

BASE_DIR = "processed"

sample_file = None

for root, _, files in os.walk(BASE_DIR):
    for f in files:
        if f.endswith(".npy"):
            sample_file = os.path.join(root, f)
            break
    if sample_file:
        break

if sample_file is None:
    print("No .npy files found in processed/")
    exit()

print("Showing spectrogram from:", sample_file)

spec = np.load(sample_file)

plt.figure(figsize=(8, 4))
plt.imshow(spec, aspect="auto", origin="lower", cmap="magma")
plt.colorbar(label="dB")
plt.title("Log-Mel Spectrogram")
plt.xlabel("Time")
plt.ylabel("Mel bands")
plt.tight_layout()
plt.show()

