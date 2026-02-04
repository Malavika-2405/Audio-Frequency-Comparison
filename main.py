import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------
# STEP 1: Load Both Audio Files
# -----------------------------
audio1, sr1 = librosa.load("audio1.wav", sr=None)
audio2, sr2 = librosa.load("audio2.wav", sr=None)

print("Audio files loaded successfully!")

# -----------------------------
# STEP 2: Convert to Frequency Domain (Spectrogram)
# -----------------------------
spec1 = np.abs(librosa.stft(audio1))
spec2 = np.abs(librosa.stft(audio2))

# -----------------------------
# STEP 3: Plot Both Spectrograms Together
# -----------------------------
plt.figure(figsize=(12, 6))

plt.subplot(2, 1, 1)
librosa.display.specshow(librosa.amplitude_to_db(spec1, ref=np.max),
                         sr=sr1, x_axis='time', y_axis='hz')
plt.colorbar()
plt.title("Frequency Spectrum - Audio 1")

plt.subplot(2, 1, 2)
librosa.display.specshow(librosa.amplitude_to_db(spec2, ref=np.max),
                         sr=sr2, x_axis='time', y_axis='hz')
plt.colorbar()
plt.title("Frequency Spectrum - Audio 2")

plt.tight_layout()
plt.show()

# -----------------------------
# STEP 4: Extract MFCC Features
# -----------------------------
mfcc1 = librosa.feature.mfcc(y=audio1, sr=sr1, n_mfcc=13)
mfcc2 = librosa.feature.mfcc(y=audio2, sr=sr2, n_mfcc=13)

# Take mean of MFCC over time
mfcc1_mean = np.mean(mfcc1, axis=1)
mfcc2_mean = np.mean(mfcc2, axis=1)

# -----------------------------
# STEP 5: Calculate Similarity Score
# -----------------------------
similarity = cosine_similarity([mfcc1_mean], [mfcc2_mean])

score = similarity[0][0] * 100

print("\n==============================")
print("Voice Similarity Score Result")
print("==============================")
print(f"Similarity Score: {score:.2f}%")

if score > 80:
    print("✅ Very Similar Voices")
elif score > 50:
    print("⚠ Moderately Similar Voices")
else:
    print("❌ Voices are Different")
