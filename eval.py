import io
import os
import argparse

import numpy as np
import torch

from models.tacotron import Tacotron
from utils.audio import AudioProcessor
from utils.generic_utils import load_config
from utils.text import phoneme_to_sequence, phonemes, symbols, text_to_sequence

class Synthesizer(object):
    def load_model(self, model_path, model_config, use_cuda):
        
        self.model_file = model_path
        print(" > Loading model ...")
        print(" | > model config: ", model_config)
        print(" | > model file: ", self.model_file)
        config = load_config(model_config)
        self.config = config
        self.use_cuda = use_cuda
        self.use_phonemes = config.use_phonemes
        self.ap = AudioProcessor(**config.audio)
        
        if self.use_phonemes:
            self.input_size = len(phonemes)
            self.input_adapter = lambda sen: phoneme_to_sequence(sen, [self.config.text_cleaner], self.config.phoneme_language)
        else:
            self.input_size = len(symbols)
            self.input_adapter = lambda sen: text_to_sequence(sen, [self.config.text_cleaner])
        
        self.model = Tacotron(self.input_size, config.embedding_size, self.ap.num_freq, self.ap.num_mels, config.r)
        # load model state
        if use_cuda:
            cp = torch.load(self.model_file)
        else:
            cp = torch.load(self.model_file, map_location=lambda storage, loc: storage)
        # load the model
        self.model.load_state_dict(cp['model'])
        if use_cuda:
            self.model.cuda()
        self.model.eval()
    
    def save_wav(self, wav, path):
        # wav *= 32767 / max(1e-8, np.max(np.abs(wav)))
        wav = np.array(wav)
        self.ap.save_wav(wav, path)
    
    def tts(self, text):
        wavs = []
        for sen in text.split('.'):
            if len(sen) < 3:
                continue
            sen = sen.strip()
            sen += '.'
            print(sen)
            sen = sen.strip()
            
            seq = np.array(self.input_adapter(sen))
            
            chars_var = torch.from_numpy(seq).unsqueeze(0).long()
            if self.use_cuda:
                chars_var = chars_var.cuda()
            mel_out, linear_out, alignments, stop_tokens = self.model.forward(
                chars_var)
            linear_out = linear_out[0].data.cpu().numpy()
            wav = self.ap.inv_spectrogram(linear_out.T)
            wavs += list(wav)
            wavs += [0] * 10000
        
        return wavs



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_path',
        type=str,
        help='Path to model file.',
        default=0)
    parser.add_argument(
        '--config_path',
        type=str,
        help='Path to config file for training.',
    )
    parser.add_argument(
        '--use_cuda',
        type=bool,
        help='Should use cuda?',
        default = False
    )

    parser.add_argument(
        'string',
        type=str,
        help='String to read'
    )
    
    args = parser.parse_args()
    synthesizer = Synthesizer()
    
    synthesizer.load_model(args.model_path,
                       args.config_path, args.use_cuda)
    
    wav = synthesizer.tts(args.string)
    synthesizer.ap.save_wav(wav, 'test.wav')