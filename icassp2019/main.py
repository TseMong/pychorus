import librosa
import numpy as np

if __name__ == '__main__':
    y, sr = librosa.load('C:\\Users\\MZ\\Desktop\\Blackbox\\the beatles mp3\\1963 Please Please Me\\01 - I Saw Her Standing There.mp3', sr=None)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
    print(S.shape, sr)
