from tqdm import tqdm
import librosa, os, json
import numpy as np

def get_beats(y, sr):
    '''
    args:   y, sr
    output: if have no beats label, will use this function
            beats_nums timestamps list.
    '''
    #y, sr = librosa.load(filepath)
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    return list(beats)

def get_melScale_logSpectrogram(y, sr):
    '''
    args:   y, sr
    output: times * 128 np.array
    '''
    def normalize(X):
        X = X.reshape((128, -1))
        means = X.mean(axis=1)
        stds = X.std(axis= 1, ddof=1)
        X= X - means[:, np.newaxis]
        X= X / stds[:, np.newaxis]
        return X.reshape((-1, 128))
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
    S = np.transpose(np.log(1+10000*S))
    return normalize(S)
    #return S

def get_beats_msls(filepath, beats=[]):   #   read beats from annotations

    y, sr = librosa.load(filepath)  #   just load once
    assert sr == 22050              #   just assert once
    if beats == []:                 #   calculate beats by self
        beats = get_beats(y, sr)
    msls = get_melScale_logSpectrogram(y, sr)

    #   we get np.array of 3 * 128, (beats-1 & beats & beats+1)
    lstm_input = np.concatenate((msls[beats-1].reshape((-1,1,128)), msls[beats].reshape((-1,1,128)), msls[beats+1].reshape((-1,1,128))), axis=1)
    ###########################################
    ###                                     ###
    ###    we have not consider condition   ###
    ###    of beats-1<0 & beats+1>max_beats ###
    ###                                     ###
    ###########################################
    return lstm_input

if __name__ == '__main__':
    pass
    
