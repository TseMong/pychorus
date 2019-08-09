from __future__ import division

import argparse, os, codecs, re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from pychorus.helpers import *

####

no_word = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '[', ']', ',', '.', '<', '>', ':']
root = 'data'
songs_root = os.path.join(root, 'songs')
accs_root = os.path.join(root, 'accompany')
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

def get_chroma_similarity(filepath: str, lyric_info: np.array) -> np.array:
    chroma, y, sr, song_length_sec = create_chroma(filepath)
    num_samples = chroma.shape[1]
    chroma_sr = num_samples / song_length_sec
    chroma_lyric = [chroma[:, int(float(lyric_info[idx, 0]) * chroma_sr): int(float(lyric_info[idx, 1]) * chroma_sr)] for idx in range(lyric_info.shape[0])]
    similarity_chroma = np.zeros((len(chroma_lyric), len(chroma_lyric)))
    for idx1 in range(len(chroma_lyric)):
        for idx2 in range(len(chroma_lyric)):
            if abs(int(lyric_info[idx1][2]) - int(lyric_info[idx2][2])) <= 1:
                similarity_chroma[idx1, idx2] = 1 - cal_chroma_similarity(chroma_lyric[idx1], chroma_lyric[idx2])
            else:
                similarity_chroma[idx1, idx2] = 0
    return similarity_chroma

def get_similar_seg_matrix(similarity_chroma: np.array) -> np.array:
    similar_seg_matrix = np.zeros((similarity_chroma.shape[0], 5)) - 1
    for idx in range(similarity_chroma.shape[0]):
        flag = 0
        for idx_lyric in range(similarity_chroma.shape[1]):
            if similarity_chroma[idx][np.argsort(- similarity_chroma[idx])[idx_lyric]] != 0 and similarity_chroma[idx][np.argsort(- similarity_chroma[idx])[idx_lyric]] != 1:
                #f.write(str(lyric_info[np.argsort(similarity_chroma[idx])[idx_lyric]][-1])+',')
                similar_seg_matrix[idx][flag] = np.argsort(- similarity_chroma[idx])[idx_lyric]
                flag += 1
            else:
                pass
            if flag == 5:
                break
    return similar_seg_matrix

def find_seg(idx: list, similar_seg_matrix: np.array, flag_matrix: np.array) -> tuple:
    start_value = similar_seg_matrix[idx[0], idx[1]]
    flag_matrix[idx[0], idx[1]] = 1
    seg1 = [start_value,]
    seg2 = [idx[0],]
    flag = False
    for idx1 in range(idx[0]+1, similar_seg_matrix.shape[0]-1):
        if (start_value+1) in similar_seg_matrix[idx1]:
            start_value += 1
            seg1.append(start_value)
            seg2.append(idx1)
            flag = True
            flag_matrix[idx1][np.argwhere(similar_seg_matrix[idx1] == start_value)] = 1
        elif idx1 <= similar_seg_matrix.shape[0] - 2 and (start_value+2) in similar_seg_matrix[idx1+1]:
            start_value += 1
            seg1.append(start_value)
            seg2.append(idx1)
            flag = True
        else:
            break
    
    return (flag, [seg1, seg2])



def get_candidate_paragraph(similar_seg_matrix: np.array) -> np.array:
    flag_matrix = np.zeros_like(similar_seg_matrix)
    seg_candidate = []
    for idx1 in range(similar_seg_matrix.shape[0]):
        for idx2 in range(similar_seg_matrix.shape[1]):
            if flag_matrix[idx1, idx2] == 0:
                save_flag, seg = find_seg([idx1, idx2], similar_seg_matrix, flag_matrix)
                if save_flag == True:
                    seg_candidate.append(seg)
                else:
                    pass
            else:
                pass
    return seg_candidate

            

#   歌词中含有歌曲名字 副歌区间开始置信度增加
#
if __name__ == "__main__":


    songname = '十年'
    #lyric_info = get_lyric_seg(os.path.join(lyric_root, songname + '.txt'))
    for item in get_lyric_seg(os.path.join(lyric_root, songname + '.txt')):
        print(item)
    #similarity_chroma = get_chroma_similarity(os.path.join(accs_root, songname + '.mp3'), lyric_info)

    #similar_seg_matrix = get_similar_seg_matrix(similarity_chroma)
    #print(get_candidate_paragraph(similar_seg_matrix))
    
    # 相似系数（与后面相似的程度） & 包含系数（副歌句子出现的位置必定是副歌）
    # 两个系数 相互trade-off 达到nash-balance
