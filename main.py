#import chorus.get_chorus as get_chorus
from chorus.get_segment_audio_txt import main, get_lyric_seg
import os, msaf, json, jams, codecs
from tqdm import tqdm
import numpy as np

root = 'data'
songs_root = os.path.join(root, 'songs')
accs_root = os.path.join(root, 'accompany')
lyric_root = os.path.join(root, 'lyrics')
json_root = os.path.join(root, 'json')

class paragraph_parser():
    def __init__(self, lyric_info, seg_point):
        self.sentence = lyric_info[:, -1]   #   句子文本
        self.start = lyric_info[:, 0].astype(np.float)  #   每句开始
        self.end = lyric_info[:, 1].astypr(np.float)    #   每句结束
        self.duration = lyric_info[:, 2].astype(np.float)   #   当前句与上一句间隔
        self.seg_candidate = seg_point  #   msaf的分割点
    
    def get_intro(self):    #   得到前奏
        vocal_start = self.start[0]
        intro = []
        for idx in range(1, len(self.seg_candidate)):   #   msaf第一个断点为0
            if abs(self.seg_candidate[idx] - vocal_start) >= 5:
                intro.append([seg_candidate[idx-1], self.seg_candidate[idx]])
            else:
                intro.append([seg_candidate[idx-1], round(vocal_start, 3)])
        self.intro = intro
        return intro






if __name__ == '__main__':
    '''
    names = [song.split('.')[0] for song in os.listdir(songs_root)]
    songPath, accPath, lyricPath = [], [], []
    for name in names:
        songPath.append(songs_root + '/' + name + '.mp2')
        accPath.append(accs_root + '/' + name + '.mp3')
        lyricPath.append(lyric_root + '/' + name + '.txt')
    np.save('./data/chorus_audio/{}.npy'.format('30songs_highlights'), get_chorus.extract(songPath, accPath, lyricPath, save_wav=False))
    '''
    with open('30songs_seg.json', 'r') as f:
        seg = json.load(f)
    #print(seg)
    with open('highlight.json', 'r') as f:
        point = json.load(f)
    #print(point)
    lyric_info = get_lyric_seg(os.path.join(lyric_root, '不再犹豫' + '.txt'))
    for idx in range(lyric_info.shape[0]):
        print('{}\t{}\t{}\t->{}\t{}'.format(idx, round(float(lyric_info[idx][-2]), 3), round(float(lyric_info[idx][0]), 3), round(float(lyric_info[idx][1]), 3), lyric_info[idx][-1]))
    print(point['不再犹豫'], seg['不再犹豫'][np.argmin(abs(np.array(seg['不再犹豫']) - point['不再犹豫']))])
    print(seg['不再犹豫'])

    #main()
