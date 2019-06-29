from __future__ import division

import argparse
import pandas as pd
from tqdm import tqdm

from pychorus.helpers import find_and_output_chorus
from pychorus import Duration


def main(args):
    find_and_output_chorus(args.input_file, args.output_file, args.min_clip_length)


if __name__ == "__main__":
    '''
    parser = argparse.ArgumentParser(
        description="Select and output the chorus of a piece of music")
    parser.add_argument("input_file", help="Path to input audio file")
    parser.add_argument(
        "--output_file",
        default="chorus.wav",
        help="Output file")
    parser.add_argument(
        "--min_clip_length",
        default=15,
        help="Minimum length (in seconds) to be considered a chorus")

    main(parser.parse_args())
    '''
    origin_root = 'D:/Original/'
    accompany_root = 'D:/Accompony/'
    lyric_root = 'D:/Lyrics_utf8/'
    songs = pd.read_csv('final.csv', header=0)
    for idx in tqdm(songs.index[100:200]):
        input_lyric = lyric_root + str(songs.loc[idx, 'songID']) + '.txt'
        input_acc = accompany_root + str(songs.loc[idx, 'songID']) + '_2.mp3'
        input_origin = origin_root + str(songs.loc[idx, 'songID']) + '_1.mp3'
        output_origin = 'data/' + str(songs.loc[idx, 'songID']) + '_chorus_B.wav'
        pattern_candidate = Duration._compute_section(input_lyric)
        min_clip_length = 15
        find_and_output_chorus(input_acc, input_origin, output_origin, pattern_candidate,min_clip_length)