import os, re
import matplotlib.pyplot as plt
from pprint import pprint
from collections import Counter
from functools import reduce
import codecs


def minDistance(word1, word2):
    if not word1:
        return len(word2 or '') or 0
    if not word2:
        return len(word1 or '') or 0
    size1 = len(word1)
    size2 = len(word2)
    last = 0
    tmp = list(range(size2 + 1))
    value = None

    for i in range(size1):
        tmp[0] = i + 1
        last = i
        # print word1[i], last, tmp
        for j in range(size2):
            if word1[i] == word2[j]:
                value = last
            else:
                value = 1 + min(last, tmp[j], tmp[j + 1])
                # print(last, tmp[j], tmp[j + 1], value)
            last = tmp[j+1]
            tmp[j+1] = value
        # print tmp
    return value
def dim_sim(i, j):
    # return i == j
    if str(i).isdigit() and str(j).isdigit():
        return i == j
    else:
        return minDistance(i, j) < 2
def max_match(lyric_lines, type_list, init_value, idx):
    result = []
    max_len = len(lyric_lines)
    for i in range(idx + 1, max_len):
        j = 0
        k = 0
        while idx + j < i and i + k < max_len \
                and type_list[idx+j] == init_value\
                and type_list[i+k] == init_value\
                and dim_sim(lyric_lines[idx+j], lyric_lines[i+k]): # 这一行可以用dim_smi 代替 ==
            j += 1
            k += 1
        result.append([i, j])  # 匹配位置和匹配长度
    return max(result, key=lambda k:k[1]) if result else []


class ComputeSection:
    def __init__(self, lyric_path:str='', pattern='chr')->list:
        with codecs.open(lyric_path, encoding='utf8', mode='r') as f:
            self.lyric_lines = []
            self.lyric_times = []
            prev_is_lyric = 0
            for line in f:
                # print('line: ', line)
                if not prev_is_lyric and line.strip():
                    lyric = ''.join([i for i in str(line) if i.isalpha()])
                    # lyric = re.sub(r'[a-zA-Z]', '', lyric)
                    if lyric.strip():
                        self.lyric_lines.append(lyric)
                        prev_is_lyric = 1
                elif prev_is_lyric:
                    try:
                        mm, ss = (re.search(r'\[(.*)\]', line).groups()[0]).split(':')
                        ts = float(mm) * 60 + float(ss)  # 单句歌词开始时间，单位秒
                        te = ts + sum(map(int, re.findall(r'\<(.*?)\,.*?\>', line))) / 1000
                        self.lyric_times.append((ts, te))
                        assert len(self.lyric_times) == len(self.lyric_lines)
                        prev_is_lyric = 0
                    except:
                        print('lyric time error!')
                        prev_is_lyric = 0

            #for i, line in enumerate(self.lyric_lines):
            #    print(str(i) + ': ' + str(line) + ' ' + str(self.lyric_times[i]))

        self.inverse_lyric_lines = self.lyric_lines[::-1]
        self.lyric_nums = list(map(len, self.lyric_lines))
        self.inverse_lyric_nums = self.lyric_nums[::-1]
        self.pattern = pattern
        self.type_step = 100
        self.accompany_min_length = 4

    def generate_list(self, inverse_lyric_lines):
        type_counter = 0

        init_value = -1 * self.type_step
        type_list = [init_value] * len(self.lyric_lines)
        jump_list = []
        inverse_type_list = [-1 * self.type_step] * len(self.lyric_lines)
        for i in range(len(self.lyric_lines)):
            match_len = 0
            if inverse_type_list[i] < 0:
                match_res = max_match(inverse_lyric_lines, inverse_type_list, init_value, i)
                if match_res and match_res[-1] > 0:
                    match_len = match_res[-1]
                    local_step = 1
                    while match_res[-1] == match_len and type_list[match_res[0]] == -1 * self.type_step:
                        jump_list.append((len(self.lyric_lines) - sum(match_res), len(self.lyric_lines) - match_res[0] - 1))
                        inverse_type_list[match_res[0]:match_res[0] + match_res[1]] = [type_counter + local_step] * \
                                                                                      match_res[1]
                        match_res = max_match(inverse_lyric_lines, inverse_type_list, init_value, i)
                        local_step += 1
                    jump_list.append((len(self.lyric_lines) - (i + match_len), len(self.lyric_lines) - i - 1))
                    inverse_type_list[i:i + match_len] = [type_counter] * match_len
                    type_counter += self.type_step
        true_type_list = inverse_type_list[::-1]
        # 补充 type<0 的类
        idx = 0
        while idx < len(true_type_list):
            if true_type_list[idx] < 0:
                ts = idx
                while idx < len(true_type_list) and true_type_list[idx] < 0:
                    te = idx
                    idx += 1
                jump_list.append((ts, te))
            else:
                idx += 1

        return jump_list, true_type_list

    def generate_list_by_time(self):
        prev_time = 0
        diff_list = []
        for i, line in enumerate(self.lyric_lines):
            cur_time = self.lyric_times[i][0]
            if prev_time:
                diff_list.append(round(cur_time - prev_time, 6))
            prev_time = cur_time
        candidate = Counter()
        for i in range(len(diff_list)):
            candidate.update([round(diff_list[i], 1)])
            if i > 1:
                candidate.update([round(diff_list[i] + diff_list[i - 1], 1)])
            if i > 2:
                candidate.update([round(diff_list[i] + diff_list[i - 1] + diff_list[i - 2], 1)])

        intern_len = candidate.most_common(2)[0][0]
        section_list = []
        section_time_list = []
        multipul_rate = 1
        inner_max_len = 3 * multipul_rate
        for i in range(len(self.lyric_times)):
            for j in range(inner_max_len):
                if i > j:
                    if round(self.lyric_times[i][0] - self.lyric_times[i - j - 1][0], 1) == intern_len * multipul_rate:
                        section_list.append(self.lyric_lines[i - j - 1:i])
                        section_time_list.append((self.lyric_times[i - j - 1][0], self.lyric_times[i][1]))

        return section_time_list

    def generate_section_list(self):  # 获取歌曲分段以及段落标签 ——（段落开始，段落结束，类别）三元组
        jump_list, type_list = self.generate_list(self.inverse_lyric_lines)
        duration = list(map(lambda l: [self.lyric_times[l[0]][0], self.lyric_times[l[1]][1]], jump_list))
        support_list = [type_list[i[0]]//self.type_step for i in jump_list]
        duration = [list(duration[i]+[support_list[i]]) for i in range(len(support_list))]
        duration = sorted(duration, key=lambda x: x[0])
        return duration


    def get_result(self):
        if self.pattern == 'chr':  # 最大字符串匹配
            jump_list, _ = self.generate_list(self.inverse_lyric_lines)
        elif self.pattern == 'num':  # 句子长度匹配
            jump_list, _ = self.generate_list(self.inverse_lyric_nums)
        elif self.pattern == 'sec':  # 歌词时间匹配
            return self.generate_list_by_time()

        duration = map(lambda l: (self.lyric_times[l[0]][0], self.lyric_times[l[1]][1]), jump_list)
        duration = sorted(duration, key=lambda x: x[0])
        return list(duration)

    def generate_accompany_list(self, song_end_time):  # 获取歌曲全部间奏
        accompany_list = []
        duration = self.get_result()
        pre_s, pre_e = 0, 0

        for cur_s, cur_e in duration:
            if cur_s - pre_e > self.accompany_min_length:
                accompany_list.append([pre_e, cur_s])
            pre_s, pre_e = cur_s, cur_e
        if song_end_time - pre_e > self.accompany_min_length:
            accompany_list.append([pre_e, song_end_time])
        return accompany_list