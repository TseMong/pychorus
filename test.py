import chorus.get_chorus as get_chorus
import os
import numpy as np
import pandas as pd

origin_root = 'D:/Original'
accompany_root = 'D:/Accompony'
lyric_root = 'D:/Lyrics_utf8'
result_root = './data/chorus_audio'
songListFile = './data/table/final.csv'

songList = pd.read_csv(songListFile)

if __name__ == '__main__':
    #names = songList.loc[:199, 'songname']
    ids = songList.loc[:199, 'songID']

    songPath, accPath, lyricPath = [], [], []
    for idx in ids:
        songPath.append(origin_root + '/' + str(idx) + '.mp3')
        accPath.append(accompany_root + '/' + str(idx) + '.mp3')
        lyricPath.append(lyric_root + '/' + str(idx) + '.txt')
    np.save('./data/chorus_audio/{}.npy'.format('200songs_highlights'), get_chorus.extract(songPath, accPath, lyricPath, save_wav=True))
    