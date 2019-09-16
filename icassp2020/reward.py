import numpy as np
import matplotlib.pyplot as plt
import librosa

def novelty(chunk_1, chunk_2):
    '''
        changing in timbre and rhythm
    args:
            chunk1:the pre chunk before boundary
            chunk2:the late chunk after boundary
    output:
            the novelty score. np.float
    '''
    pass

def homogeneity(chunk):
    '''
    args:
            chunk:the finished chunk before boundary
    output:
            the homogeneity score. np.float
    '''
    assert chunk.shape[-1] == 128
    return np.sum(chunk.std(axis=0))

def repetition(chunk, chunk_set):
    '''
    args:
            chunk:the current finished chunk before boundary
            chun_set:the chunks before current boundary
    output:
            the repetition score. np.float
    '''
    pass

def smooth(y):
    result = np.zeros_like(y)
    for idx in range(len(y)):
        result[idx] = np.mean(y[max(idx-256, 0):min(len(y), idx+256)])
    return result


if __name__ == '__main__':
    msls = np.load('倒带.npy')
    x = librosa.core.frames_to_time(range(msls.shape[0]))
    y = smooth(msls[:, 0])
    plt.plot(x, y)
    plt.savefig('倒带.png')