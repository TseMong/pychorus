from chorus.__init__ import *
from chorus.models import MusicHighlighter
from chorus.lib import *

import tensorflow as tf
import numpy as np
import os
import pandas as pd
from tqdm import tqdm
import chorus.get_segment_only_txt as get_segment_only_txt
import chorus.get_segment_audio_txt as get_segment_audio_txt

os.environ["CUDA_VISIBLE_DEVICES"] = gpu

def extract(songPath_list: list, accPath_list: list, lyricPath_list: list, length=30, save_score=False, save_thumbnail=False, save_wav=False) -> np.array:
    '''
    input   :
        
        songPath_list  ->  song path list
        lyricPath_list ->  lyric path list
    
    
    output   :
        
        return  ->  chorus timestamp include start and end, shape is (len(songPath_list), 2)

    '''


    with tf.Session() as sess:
        model = MusicHighlighter() 
        sess.run(tf.global_variables_initializer())
        model.saver.restore(sess, './chorus/paras/model')
 
        ##
        total_highlight = []
        ##

        for idx, songPath in tqdm(enumerate(songPath_list)):

            # using acc as imput data is better

            name = os.path.split(songPath)[1].split('.')[0]
            sr = 22050 # hard code
            _, spectrogram, duration = audio_read(accPath_list[idx])
            audio, _, _ = audio_read(songPath)
            n_chunk, remainder = np.divmod(duration, 3)
            chunk_spec = chunk(spectrogram, n_chunk)
            pos = positional_encoding(batch_size=1, n_pos=n_chunk, d_pos=model.dim_feature*4)
            
            n_chunk = n_chunk.astype('int')
            chunk_spec = chunk_spec.astype('float')
            pos = pos.astype('float')
            
            attn_score = model.calculate(sess=sess, x=chunk_spec, pos_enc=pos, num_chunk=n_chunk)
            attn_score = np.repeat(attn_score, 3)
            attn_score = np.append(attn_score, np.zeros(remainder))

            # score
            attn_score = attn_score / attn_score.max()
            if save_score:
                np.save('result/{}_score.npy'.format(name), attn_score)

            # thumbnail
            attn_score = attn_score.cumsum()
            attn_score = np.append(attn_score[length], attn_score[length:] - attn_score[:-length])
            index = np.argmax(attn_score)
            
            highlight_start_dis = 2 ** 16 
            highlight_start_dis_res = []
            # distance is better than interval 

            cs = get_segment_only_txt.ComputeSection(lyricPath_list[idx], pattern='chr')
            pattern_candidate = cs.get_result() # 2-D

            #pattern_candidate = get_segment_audio_txt.get_final_paragraph(songPath, lyricPath_list[idx])


            for start, end in pattern_candidate:
                #if index >= start and index < end:
                #    highlight_interval_res = [start, end]
                if abs(index - start) < highlight_start_dis:
                    highlight_start_dis = abs(index - start)
                    highlight_start_dis_res = [start, end]
                else:
                    pass

            total_highlight.append(highlight_start_dis_res)

            if save_thumbnail:
                #np.save('result/{}_highlight_interval.npy'.format(name), highlight_interval_res)
                np.save('result/{}_highlight_distance.npy'.format(name), highlight_start_dis_res)

            if save_wav:
                #librosa.output.write_wav('result/{}_audio_int.wav'.format(name), audio[int(highlight_interval_res[0]*sr):int(highlight_interval_res[1]*sr)], sr)
                librosa.output.write_wav('./data/chorus_audio_only_txt/{}.wav'.format(name), audio[int(highlight_start_dis_res[0]*sr):int(highlight_start_dis_res[1]*sr)], sr)
    if len(songPath_list) == 1:
        return np.array(total_highlight).reshape(1, 2)[0]
    else:
        return np.array(total_highlight).reshape(len(songPath_list), 2)

if __name__ == '__main__':
    pass

