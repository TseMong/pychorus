import os
import re
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
                and dim_sim(lyric_lines[idx+j],lyric_lines[i+k]): # 这一行可以用dim_smi 代替 ==
            j += 1
            k += 1
        result.append([i, j])  # 匹配位置和匹配长度
    return max(result, key=lambda k:k[1]) if result else []



def _compute_section(lyric_path:str='')->list:
    with codecs.open(lyric_path, encoding='utf8', mode='r') as f:
        lyric_lines = []
        lyric_times = []
        prev_is_lyric = 0
        for line in f:
            # print('line: ', line)
            if not prev_is_lyric and line.strip():
                lyric = ''.join([i for i in str(line) if i.isalpha()])
                if lyric.strip():
                    lyric_lines.append(lyric)
                    prev_is_lyric = 1
            elif prev_is_lyric:
                try:
                    mm, ss = (re.search(r'\[(.*)\]', line).groups()[0]).split(':')
                    ts = float(mm) * 60 + float(ss)  # 单句歌词开始时间，单位秒
                    te = ts + sum(map(int, re.findall(r'\<(.*?)\,.*?\>', line))) / 1000
                    lyric_times.append((ts, te))
                    assert len(lyric_times) == len(lyric_lines)
                    prev_is_lyric = 0
                except:
                    print('lyric time error!')
                    prev_is_lyric = 0

    type_counter = 0
    type_step = 10
    init_value = -1 * type_step
    type_list = [init_value] * len(lyric_lines)
    jump_list = []
    inverse_type_list = [-1 * type_step] * len(lyric_lines)
    inverse_lyric_lines = lyric_lines[::-1]


    for i in range(len(lyric_lines)):
        match_len = 0
        if inverse_type_list[i] < 0:
            match_res = max_match(inverse_lyric_lines, inverse_type_list, init_value, i)
            if match_res and match_res[-1] > 0:
                match_len = match_res[-1]
                local_step = 1
                while match_res[-1] == match_len and type_list[match_res[0]] == -1 * type_step:
                    jump_list.append((len(lyric_lines) - sum(match_res), len(lyric_lines) - match_res[0] - 1))
                    inverse_type_list[match_res[0]:match_res[0] + match_res[1]] = [type_counter + local_step] * \
                                                                                  match_res[1]
                    match_res = max_match(inverse_lyric_lines, inverse_type_list, init_value, i)
                    local_step += 1
                jump_list.append((len(lyric_lines) - (i + match_len), len(lyric_lines) - i - 1))
                inverse_type_list[i:i + match_len] = [type_counter] * match_len
                type_counter += type_step

    duration = map(lambda l:(lyric_times[l[0]][0], lyric_times[l[1]][1]), jump_list)
    return sorted(list(duration), key=lambda x:x[0])

