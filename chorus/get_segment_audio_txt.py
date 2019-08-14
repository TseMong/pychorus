from __future__ import division

import argparse, os, codecs, re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from chorus.helpers import *

####

no_word = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '[', ']', ',', '.', '<', '>', ':']
root = 'data'
candidate_limit = 10
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
            if abs(int(lyric_info[idx1][2]) - int(lyric_info[idx2][2])) == 0:   # word num error
                similarity_chroma[idx1, idx2] = 1 - cal_chroma_similarity(chroma_lyric[idx1], chroma_lyric[idx2])
            else:
                similarity_chroma[idx1, idx2] = 0
    return similarity_chroma

def get_similar_seg_matrix(similarity_chroma: np.array) -> np.array:
    similar_seg_matrix = np.zeros((similarity_chroma.shape[0], 10)) - 1
    for idx in range(similarity_chroma.shape[0]):
        flag = 0
        for idx_lyric in range(similarity_chroma.shape[1]):
            current_idx = np.argsort(- similarity_chroma[idx])[idx_lyric]
            if similarity_chroma[idx][current_idx] not in [0, 1] and similarity_chroma[idx][current_idx] >= 0.9:
                #f.write(str(lyric_info[np.argsort(similarity_chroma[idx])[idx_lyric]][-1])+',')
                similar_seg_matrix[idx][flag] = current_idx
                flag += 1
            else:
                pass
            if flag == candidate_limit or similarity_chroma[idx][current_idx] < 0.9 :  #   candidate_limit 相似
                break
    return similar_seg_matrix

def find_seg(idx: list, similar_seg_matrix: np.array, flag_matrix: np.array) -> tuple:
    start_value = int(similar_seg_matrix[idx[0], idx[1]])
    if start_value == -1:
        return (False, [[], []])
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
    if len(seg1) <= 2:  # or len(seg1) >= 20:
        flag = False
    
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

def get_root_lyric_line(candidate_paragraph: list, sentence_num: int) -> list:
    def add_node(node_list: np.array, row: int,node: int) -> np.array:
        for idx in range(node_list[row].shape[-1]):
            if node in node_list[row] or node == row: # node == row
                break
            elif node_list[row][idx] == -1:
                node_list[row][idx] = node
            else:
                pass
        return node_list[row]

    def get_min_index(similar_list: np.array, idx: int) -> int:
        # not all is -1
        # not the min index
        while max(similar_list[idx]) >= 0 and min([int(num) for num in similar_list[idx] if num >= 0]) < idx:
            idx = min([int(num) for num in similar_list[idx] if num >= 0])

        return int(idx)

    similar_list = np.zeros((sentence_num, candidate_limit)) - 1
    for seg_group in candidate_paragraph:
        seg1, seg2 = seg_group
        for idx in range(len(seg1)):
            similar_list[seg2[idx]] = add_node(similar_list, seg2[idx], seg1[idx])

    #for idx in range(sentence_num-1, -1, -1):
        #if idx not in similar_list[get_min_index(similar_list, idx)]:
        #    similar_list[get_min_index(similar_list, idx)] = add_node(similar_list, get_min_index(similar_list, idx), idx)
        #similar_list[idx] = add_node(similar_list, idx, idx) # add idx for every line 
    
    return similar_list

def get_seg_point(root_lyric_line: np.array, lyric_info: np.array) -> np.array: #   将问题规约为 “每一句属于上一段还是下一段，或者不切分”
    seg_point = np.zeros(root_lyric_line.shape[0])
    seg_point[0] = 1
    def get_break_and_cont(first_line: np.array, second_line: np.array) -> int:   #   获取上一句间断点
        break_num = 0   #   上下句的间断
        cont_num = 0    #   上下句连续
        for point in first_line:
            if point == -1:
                break
            elif point + 1 in second_line:
                cont_num += 1
            else:
                break_num += 1
        for point in second_line:
            if point == -1:
                break
            elif point - 1 in first_line and point != 0:
                pass
            else:
                break_num += 1
        return break_num, cont_num

    for idx in range(1, root_lyric_line.shape[0]-1):    #   判断当前句属于上段还是下段,又或者当前不切分-----当前句是否切分
        prev_break_num, prev_cont_num = get_break_and_cont(root_lyric_line[idx-1], root_lyric_line[idx])
        #next_break_num, next_cont_num = get_break_and_cont(root_lyric_line[idx], root_lyric_line[idx+1])
        if float(lyric_info[idx, -2]) >= 5: #   歌曲段落间隔大于5s 切分， 另开新的一段
            seg_point[idx] = 1
            if seg_point[idx-1] == 1:   #   连续分割track
                seg_point[idx-1] = 0
        elif sum(root_lyric_line[idx]) == -10:  #   如果[-1, ..., -1]，则自动与上一句连接
            continue
        elif prev_break_num - prev_cont_num > 0:
            seg_point[idx] = 1
            if seg_point[idx-1] == 1 and idx != 1:   #   连续分割track, 开头避开
                seg_point[idx-1] = 0
            else:
                pass
        else:
            pass
    
        #if (prev_break_num - prev_cont_num) - (next_break_num - next_cont_num) >= 0:
        #    seg_point[]

    return seg_point

def get_final_paragraph(songPath: str, lyricPath: str) -> list: #   Origin song , not accompany
    lyric_info = get_lyric_seg(lyricPath)
    similarity_chroma = get_chroma_similarity(songPath, lyric_info)
    similar_seg_matrix = get_similar_seg_matrix(similarity_chroma)
    root_line = get_root_lyric_line(get_candidate_paragraph(similar_seg_matrix), similar_seg_matrix.shape[0])
    seg_point = get_seg_point(root_line, lyric_info)
    final_paragraph = []
    temp_paragraph = []
    for idx, point in enumerate(seg_point):
        if point == 1 and idx == 0: #   开头为1
            temp_paragraph = [float(lyric_info[idx, 0]), float(lyric_info[idx, 1])]
        elif point == 1 and idx != len(seg_point) - 1:  #   歌中的1，不是末尾的1，不是开头1
            temp_paragraph[1] = float(lyric_info[idx-1, 1])
            final_paragraph.append(temp_paragraph)
            temp_paragraph = [float(lyric_info[idx, 0]), float(lyric_info[idx, 1])]
        elif point == 1 and idx == len(seg_point) - 1:  #   末尾的1
            temp_paragraph[1] = float(lyric_info[idx-1, 1])
            final_paragraph.append(temp_paragraph)
            temp_paragraph = [float(lyric_info[idx, 0]), float(lyric_info[idx, 1])]
            final_paragraph.append(temp_paragraph)
        elif idx == len(seg_point) - 1: #   末尾0
            temp_paragraph[1] = float(lyric_info[idx, 1])
            final_paragraph.append(temp_paragraph)
        else:   #   中间0，非开头0，非末尾0
            temp_paragraph[1] = float(lyric_info[idx, 1])
    return final_paragraph
    






#   歌词中含有歌曲名字 副歌区间开始置信度增加
#
if __name__ == "__main__":


    songname = '爱'   # 因为爱情
    lyric_info = get_lyric_seg(os.path.join(lyric_root, songname + '.txt'))

    #similarity_chroma = get_chroma_similarity(os.path.join(accs_root, songname + '.mp3'), lyric_info)
    similarity_chroma = get_chroma_similarity(os.path.join(songs_root, songname + '.mp2'), lyric_info)

    similar_seg_matrix = get_similar_seg_matrix(similarity_chroma)
    #for idx in range(similar_seg_matrix.shape[0]):
    #    print('{}\t{}'.format(lyric_info[idx][-1], [lyric_info[int(index)][-1] for index in similar_seg_matrix[idx] if index >= 0]))    
    #    print('{}\t{}'.format(similarity_chroma[idx][idx], [round(similarity_chroma[idx, int(index)], 3) for index in similar_seg_matrix[idx] if index >= 0]))
    root_line = get_root_lyric_line(get_candidate_paragraph(similar_seg_matrix), similar_seg_matrix.shape[0])
    seg_point = get_seg_point(root_line, lyric_info)
    for idx, point in enumerate(seg_point):
        print('{}\t{}\t{}\t{}\t->{}\t{}'.format(root_line[idx], point, round(float(lyric_info[idx][-2]), 3), round(float(lyric_info[idx][0]), 3), round(float(lyric_info[idx][1]), 3), lyric_info[idx][-1]))
    print(get_final_paragraph(os.path.join(songs_root, songname + '.mp2'), os.path.join(lyric_root, songname + '.txt')))
    # 相似系数（与后面相似的程度） & 包含系数（副歌句子出现的位置必定是副歌）
    # 两个系数 相互trade-off 达到nash-balance
    # 部分地方整句拆开  脑袋都是你心里都是你 脑袋都是你/心里都是你
    
    
    # 相似矩阵的最大索引越界
