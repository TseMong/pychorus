from __future__ import print_function

import numpy as np
import scipy
import matplotlib.pyplot as plt

import sklearn.cluster

import librosa
import librosa.display
import os

################################################################################
# load file
def load_file():
    file_path = os.path.join(os.path.abspath('.'), "BEYOND-不再犹豫-drum.mp3")
    y, sr = librosa.load(file_path)
    return y, sr



if __name__ == "__main__":
    y, sr = load_file()
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    o_env = librosa.onset.onset_strength(y, sr=sr)
    times = librosa.frames_to_time(np.arange(len(o_env)), sr=sr)
    onset_frames = librosa.onset.onset_detect(onset_envelope=o_env, sr=sr)
    with open('beats.txt', 'w') as f:
        for time in times[onset_frames]:
            f.writelines(str(time) + '\n')
    # import matplotlib.pyplot as plt
    # D = np.abs(librosa.stft(y))
    # plt.figure()
    # ax1 = plt.subplot(2, 1, 1)
    # librosa.display.specshow(librosa.amplitude_to_db(D, ref=np.max),
    #                      x_axis='time', y_axis='log')
    # plt.title('Power spectrogram')
    # plt.subplot(2, 1, 2, sharex=ax1)
    # plt.plot(times, o_env, label='Onset strength')
    # plt.vlines(times[onset_frames], 0, o_env.max(), color='r', alpha=0.9,
    #            linestyle='--', label='Onsets')
    # plt.axis('tight')
    # plt.legend(frameon=True, framealpha=0.75)
    # plt.show()