import os
import argparse
import torch
import importlib
import numpy as np
import scipy
import re

#from nltk.tokenize import sent_tokenize
#from spacy.lang.en import English
from TTS.server.synthesizer import Synthesizer

import nltk
from nltk.corpus.reader.util import *


if __name__ == "__main__":

    def convert_boolean(x):
        return x.lower() in ['true', '1', 'yes']

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input',
        type=str,
        help='Path to input text file.',
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Path to save final wav file.',
    )
    parser.add_argument('--tts_checkpoint', type=str, help='path to TTS checkpoint file')
    parser.add_argument('--tts_config', type=str, help='path to TTS config.json file')
    parser.add_argument('--tts_speakers', type=str, help='path to JSON file containing speaker ids, if speaker ids are used in the model')
    parser.add_argument('--wavernn_lib_path', type=str, default=None, help='path to WaveRNN project folder to be imported. If this is not passed, model uses Griffin-Lim for synthesis.')
    parser.add_argument('--wavernn_file', type=str, default=None, help='path to WaveRNN checkpoint file.')
    parser.add_argument('--wavernn_config', type=str, default=None, help='path to WaveRNN config file.')
    parser.add_argument('--use_cuda', type=convert_boolean, default=True, help='true to use CUDA.')
    parser.add_argument('--is_wavernn_batched', type=convert_boolean, default=True, help='true to use batched WaveRNN.')
    parser.add_argument('--pwgan_lib_path', type=str, default=None, help='path to ParallelWaveGAN project folder to be imported. If this is not passed, model uses Griffin-Lim for synthesis.')
    parser.add_argument('--pwgan_file', type=str, default=None, help='path to ParallelWaveGAN checkpoint file.')
    parser.add_argument('--pwgan_config', type=str, default=None, help='path to ParallelWaveGAN config file.')

    args = parser.parse_args()

    wavernn_checkpoint_file = "pretrained_models/WaveRNN_22k/best_model.pth.tar"
    wavernn_config_file = "pretrained_models/WaveRNN_22k/config.json"
    tts_config_file = "pretrained_models/tactron2_graves/config.json"
    tts_checkpoint_file = "pretrained_models/tactron2_graves/best_model.pth.tar"

    if not args.tts_checkpoint and os.path.isfile(tts_checkpoint_file):
        args.tts_checkpoint = tts_checkpoint_file
    if not args.tts_config and os.path.isfile(tts_config_file):
        args.tts_config = tts_config_file
    if not args.wavernn_file and os.path.isfile(wavernn_checkpoint_file):
        args.wavernn_file = wavernn_checkpoint_file
    if not args.wavernn_config and os.path.isfile(wavernn_config_file):
        args.wavernn_config = wavernn_config_file

    synthesizer = Synthesizer(args)

    try:
        path = os.path.realpath(os.path.dirname(__file__))
    except NameError as e:
        path = './'

    with open(args.input,'r') as f:
        txt = f.read()
    cleared_txt = re.sub('\n{2,}', '\n\n', txt)
    paragraphs = cleared_txt.split('\n\n')

    for i, p in enumerate(paragraphs):
        input_text = p.replace('\n', ' ')
        wav = synthesizer.tts(input_text)
        with open('%s/%d.wav'%(args.output, i+1), 'wb') as fout:
            fout.write(wav.read())




