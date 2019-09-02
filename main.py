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
    def __init__(self, lyric_info, seg_point, chorus_pre):
        self.sentence = lyric_info[:, -1]   #   句子文本
        self.start = lyric_info[:, 0].astype(np.float)  #   每句开始
        self.end = lyric_info[:, 1].astype(np.float)    #   每句结束
        self.duration = lyric_info[:, 3].astype(np.float)   #   当前句与上一句间隔
        self.seg_candidate = seg_point  #   msaf的分割点
        self.chorus_pre = chorus_pre
    
    def get_intro(self):    #   得到前奏    begin_silence = intro[0]
        vocal_start = self.start[0]
        intro = []
        for idx in range(1, len(self.seg_candidate)):   #   msaf第一个断点为0
            if - (self.seg_candidate[idx] - vocal_start) >= 3:
                intro.append([self.seg_candidate[idx-1], self.seg_candidate[idx]])
            else:
                intro.append([self.seg_candidate[idx-1], round(vocal_start, 3)])
                break
        self.intro = np.array(intro)
        return self.intro

    def get_reintro(self):  #   得到间奏
        idx = np.argwhere(self.duration > 5)
        if idx[0] == 0:
            idx = idx[1:]
        reintro = np.vstack((self.end[idx-1].reshape(-1,1), self.start[idx].reshape(-1,1)))
        while reintro[-1, 1] > self.end[-1]:
            reintro = reintro[:-1]
        self.reintro = reintro.reshape(-1, 2)
        return self.reintro

    def get_outro(self):    #   得到尾奏
        vocal_end = self.end[-1]
        outro = []
        for idx in range(1, len(self.seg_candidate)):   #   msaf第一个断点为0
            if self.seg_candidate[idx] - vocal_end >= 3 and self.seg_candidate[idx-1] < vocal_end:
                outro.append([vocal_end, self.seg_candidate[idx]])
            elif self.seg_candidate[idx] - vocal_end >= 3:
                outro.append([self.seg_candidate[idx-1], self.seg_candidate[idx]])
            else:
                pass
        self.outro = np.array(outro)
        return self.outro

    def get_candidate_seg_sentence(self):   #   得到分割句子
        seg_candidate = self.seg_candidate.copy()
        seg_points = np.zeros((len(self.start),))
        seg_points[0] = 1
        start_point = seg_candidate.pop(0)  #   掐头
        while start_point < self.start[0] + 2:
            start_point = seg_candidate.pop(0)
        seg_candidate.insert(0, start_point)
        end_point = seg_candidate.pop()    #   去尾
        while end_point > self.end[-1] - 2:
            end_point = seg_candidate.pop()
        seg_candidate.insert(-1, end_point)
        for point in seg_candidate:
            for idx in range(len(self.start)):
                if self.start[idx] < point < self.end[idx]:
                    if idx < len(self.start) - 1 and self.duration[idx] > self.duration[idx+1]: 
                        seg_points[idx] = 1
                    elif idx < len(self.start) - 1:
                        seg_points[idx+1] = 1
                    else:
                        pass
                elif idx < len(self.start) - 1 and self.end[idx] <= point <= self.start[idx+1]:
                    seg_points[idx+1] = 1
                else:
                    pass
        
        return seg_points




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
    with codecs.open('30songs_seg.json', mode='r', encoding='GBK') as f:
        seg = json.load(f)
    #print(seg)
    with codecs.open('highlight.json', mode='r', encoding='GBK') as f:
        point = json.load(f)
    # #print(point)
    lyric_info = get_lyric_seg(os.path.join(lyric_root, '大海' + '.txt'))
    #print(lyric_info)
    pp = paragraph_parser(lyric_info, seg['大海'], point['大海'])
    seg_points = pp.get_candidate_seg_sentence()
    for idx in range(lyric_info.shape[0]):
        print('{}\t{}\t{}\t->{}\t{}\t\t{}'.format(idx, round(float(lyric_info[idx][-2]), 3), round(float(lyric_info[idx][0]), 3), round(float(lyric_info[idx][1]), 3), seg_points[idx],lyric_info[idx][-1]))
    print(point['大海'], seg['大海'][np.argmin(abs(np.array(seg['大海']) - point['大海']))])
    print(seg['大海'])

    #main()
