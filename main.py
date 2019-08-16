import chorus.get_chorus as get_chorus
from chorus.get_segment_audio_txt import main
import os
import numpy as np

root = 'data'
songs_root = os.path.join(root, 'songs')
accs_root = os.path.join(root, 'accompany')
lyric_root = os.path.join(root, 'lyrics')
json_root = os.path.join(root, 'json')

if __name__ == '__main__':
    '''
    names = [song.split('.')[0] for song in os.listdir(songs_root)]
    songPath, accPath, lyricPath = [], [], []
    for name in names:
        songPath.append(songs_root + '/' + name + '.mp2')
        accPath.append(accs_root + '/' + name + '.mp3')
        lyricPath.append(lyric_root + '/' + name + '.txt')
    np.save('./data/chorus_audio/{}.npy'.format('30songs_highlights'), get_chorus.extract(songPath, accPath, lyricPath, save_wav=True))
    '''
    
    main()
