from __future__ import division

import argparse, os, codecs, re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from pychorus.helpers import *
from pychorus import Duration

####

no_word = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '[', ']', ',', '.', '<', '>', ':']
root = 'data'
songs_root = os.path.join(root, 'songs')
lyric_root = os.path.join(root, 'lyrics')
json_root = os.path.join(root, 'json')

####


def get_sentence_ts(line: str) -> float:
    minute, second = re.search(r'\[(.*)\]', line).groups()[0].split(':')
    return int(minute) * 60 + float(second)
    
def get_word_ts(line: str) -> list:
    second = re.findall(r'\<(.*?)\,.*?\>', line)
    second = [int(sec) for sec in second]
    return second

def get_words(line: str) -> str:
    for chr in no_word:
        line = line.replace(chr, '')
    return line.strip()

def cal_lyric_similarity(sentence1: str, sentence2: str) -> float:
    ''' content and position '''
    '''两个句子相同字符数 / 长句子长度 * （短句子 / 长句子）'''

    sentence1, sentence2 = list(sentence1), list(sentence2)
    if len(sentence1) > len(sentence2):
        return round(sum([1 for word in sentence2 if word in sentence1]) / len(sentence1) * len(sentence2) / len(sentence1), 3)
    else:
        return round(sum([1 for word in sentence1 if word in sentence2]) / len(sentence2) * len(sentence1) / len(sentence2), 3)

def cal_chroma_similarity(chroma1: np.array, chroma2: np.array) -> float:
    '''计算以单句歌词为界的音频相似度'''
    def cal_iou_smilarity(chroma1: np.array, chroma2: np.array) -> float:
        assert chroma1.shape >= chroma2.shape
        return np.array([((chroma2 - chroma1[:, idx:idx+chroma2.shape[1]]) ** 2).mean() for idx in range(chroma1.shape[1] - chroma2.shape[1] + 1)]).min()

    if chroma1.shape[-1] >= chroma2.shape[-1]:
        return cal_iou_smilarity(chroma1, chroma2)
    else:
        return cal_iou_smilarity(chroma2, chroma1)

def get_lyric_seg(filepath: str) -> tuple:
    '''
    input:
    filepath: string
    
    output:
    info   : tuple (starttime, endtime, word_num_per_sentence, re-intro, sentence)

    '''

    with codecs.open(filepath, mode='r', encoding='utf8') as f:
        sentence = []
        sentence_ts = []
        sentence_duration = []
        for line in f:
            if line.strip() == '':
                continue
            else:
                sentence.append(get_words(line))
                sentence_ts.append(get_sentence_ts(line))
                sentence_duration.append(sum(get_word_ts(line)))

    sentence =  [sec for sec in sentence if sec != '']
    sentence_duration = [ts/1000 for ts in sentence_duration if ts != 0]
    sentence_start = sorted(list(set(sentence_ts)))
    word_num = [len(sec) for sec in sentence]
    sentence_end = [sentence_start[idx] + sentence_duration[idx] for idx in range(len(sentence))]
    re_intro = [sentence_start[0]] + [sentence_start[idx] - sentence_end[idx-1] for idx in range(1, len(sentence))]
    assert len(sentence) == len(sentence_end) == len(sentence_start) == len(word_num) == len(re_intro)

    info = [[sentence_start[idx], sentence_end[idx], word_num[idx], re_intro[idx], sentence[idx]] for idx in range(len(sentence))]

    return np.array(info)

#   歌词中含有歌曲名字 副歌区间开始置信度增加
#   

if __name__ == "__main__":

    #for item in get_lyric_seg(os.path.join(lyric_root, '7022619.txt')):
    #    print(item)

    lyric_info = get_lyric_seg(os.path.join(lyric_root, '7022619.txt'))
    #interval = np.array(lyric_info[:, 3], dtype=float)
    #interval = [round(inv, 3) for inv in interval if inv <= (np.mean(interval) + np.std(interval) * 3)]
    #interval = [round(inv, 3) for inv in interval if inv >= (np.mean(interval) + np.std(interval) * 1)]
    #print(interval)

    chroma, y, sr, song_length_sec = create_chroma(os.path.join(songs_root, '十年.mp2'))
    num_samples = chroma.shape[1]


    # Denoise the time lag matrix
    chroma_sr = num_samples / song_length_sec
    chroma_lyric = [chroma[:, int(float(lyric_info[idx, 0]) * chroma_sr): int(float(lyric_info[idx, 1]) * chroma_sr)] for idx in range(lyric_info.shape[0])]
    for idx in range(len(chroma_lyric)):
        print(lyric_info[14][-1], lyric_info[idx][-1], cal_chroma_similarity(chroma_lyric[14], chroma_lyric[idx]))