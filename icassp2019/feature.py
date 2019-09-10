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
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
    return np.transpose(np.log(1+10000*S))

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
    root = 'C:/Users/MZ/Desktop/Blackbox/The beatles'
    audio_dir = os.path.join(root, 'audios')
    annotation_dir = os.path.join(root, 'annotations')
    filepool = os.listdir('C:/Users/MZ/Desktop/Blackbox/The beatles/mslms')
    for file in tqdm(os.listdir(annotation_dir)):
        #if file.split('.')[0] + '.npy' in filepool:
        #    continue
        anno = json.load(open(os.path.join(annotation_dir, file), 'r'))
        beats = np.array(anno['beats'], dtype=np.float)
        beats = librosa.core.time_to_frames(beats, sr=22050, hop_length=512)#, n_fft=2048)
        np.save('C:/Users/MZ/Desktop/Blackbox/The beatles/mslms/{}.npy'.format(file.split('.')[0]), get_beats_msls(os.path.join(audio_dir, file.split('.')[0] + '.mp3'), beats))
        # with open(os.path.join(annotation_dir, file), 'w') as f:
        #     json.dump(anno, f, indent=2)
