import io
import re
import sys

import numpy as np
import torch
import yaml

from TTS.utils.audio import AudioProcessor
from TTS.utils.generic_utils import load_config, setup_model
from TTS.utils.speakers import load_speaker_mapping
# pylint: disable=unused-wildcard-import
# pylint: disable=wildcard-import
from TTS.utils.synthesis import *

from TTS.utils.text import make_symbols, phonemes, symbols

import scipy as sp


alphabets = r"([A-Za-z])"
prefixes = r"(Mr|St|Mrs|Ms|Dr)[.]"
suffixes = r"(Inc|Ltd|Jr|Sr|Co)"
starters = r"(Mr|Mrs|Ms|Dr|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
acronyms = r"([A-Z][.][A-Z][.](?:[A-Z][.])?)"
websites = r"[.](com|net|org|io|gov)"


class Synthesizer(object):
    def __init__(self, config):
        self.wavernn = None
        self.pwgan = None
        self.config = config
        self.use_cuda = self.config.use_cuda
        if self.use_cuda:
            assert torch.cuda.is_available(), "CUDA is not availabe on this machine."
        self.load_tts(self.config.tts_checkpoint, self.config.tts_config,
                      self.config.use_cuda)
        if self.config.wavernn_lib_path:
            self.load_wavernn(self.config.wavernn_lib_path, self.config.wavernn_file,
                              self.config.wavernn_config, self.config.use_cuda)
        if self.config.pwgan_lib_path:
            self.load_pwgan(self.config.pwgan_lib_path, self.config.pwgan_file,
                            self.config.pwgan_config, self.config.use_cuda)

        self.sos = sp.signal.butter(10, .9*self.tts_config.audio['mel_fmax'], btype='low', fs=self.tts_config.audio['sample_rate'], output='sos')
        self.sos1 = sp.signal.butter(10, 150, btype='high', fs=self.tts_config.audio['sample_rate'], output='sos')

    def load_tts(self, tts_checkpoint, tts_config, use_cuda):
        # pylint: disable=global-statement
        global symbols, phonemes

        print(" > Loading TTS model ...")
        print(" | > model config: ", tts_config)
        print(" | > checkpoint file: ", tts_checkpoint)

        self.tts_config = load_config(tts_config)
        self.use_phonemes = self.tts_config.use_phonemes
        self.ap = AudioProcessor(**self.tts_config.audio)

        if 'characters' in self.tts_config.keys():
            symbols, phonemes = make_symbols(**self.tts_config.characters)

        if self.use_phonemes:
            self.input_size = len(phonemes)
        else:
            self.input_size = len(symbols)
        # TODO: fix this for multi-speaker model - load speakers
        if self.config.tts_speakers is not None:
            self.tts_speakers = load_speaker_mapping(self.config.tts_speakers)
            num_speakers = len(self.tts_speakers)
        else:
            num_speakers = 0
        self.tts_model = setup_model(self.input_size, num_speakers=num_speakers, c=self.tts_config)
        # load model state
        cp = torch.load(tts_checkpoint, map_location=torch.device('cpu'))
        # load the model
        self.tts_model.load_state_dict(cp['model'])
        if use_cuda:
            self.tts_model.cuda()
        self.tts_model.eval()
        self.tts_model.decoder.max_decoder_steps = 3000
        if 'r' in cp:
            self.tts_model.decoder.set_r(cp['r'])

    def load_wavernn(self, lib_path, model_file, model_config, use_cuda):
        # TODO: set a function in wavernn code base for model setup and call it here.
        sys.path.append(lib_path) # set this if WaveRNN is not installed globally
        #pylint: disable=import-outside-toplevel
        from WaveRNN.models.wavernn import Model
        print(" > Loading WaveRNN model ...")
        print(" | > model config: ", model_config)
        print(" | > model file: ", model_file)
        self.wavernn_config = load_config(model_config)
        # This is the default architecture we use for our models.
        # You might need to update it
        self.wavernn = Model(
            rnn_dims=512,
            fc_dims=512,
            mode=self.wavernn_config.mode,
            mulaw=self.wavernn_config.mulaw,
            pad=self.wavernn_config.pad,
            use_aux_net=self.wavernn_config.use_aux_net,
            use_upsample_net=self.wavernn_config.use_upsample_net,
            upsample_factors=self.wavernn_config.upsample_factors,
            feat_dims=80,
            compute_dims=128,
            res_out_dims=128,
            res_blocks=10,
            hop_length=self.ap.hop_length,
            sample_rate=self.ap.sample_rate,
        ).cuda()

        check = torch.load(model_file, map_location="cpu")
        self.wavernn.load_state_dict(check['model'])
        if use_cuda:
            self.wavernn.cuda()
        self.wavernn.eval()

    def load_pwgan(self, lib_path, model_file, model_config, use_cuda):
        sys.path.append(lib_path) # set this if ParallelWaveGAN is not installed globally
        #pylint: disable=import-outside-toplevel
        from parallel_wavegan.models import ParallelWaveGANGenerator
        print(" > Loading PWGAN model ...")
        print(" | > model config: ", model_config)
        print(" | > model file: ", model_file)
        with open(model_config) as f:
            self.pwgan_config = yaml.load(f, Loader=yaml.Loader)
        self.pwgan = ParallelWaveGANGenerator(**self.pwgan_config["generator_params"])
        self.pwgan.load_state_dict(torch.load(model_file, map_location="cpu")["model"]["generator"])
        self.pwgan.remove_weight_norm()
        if use_cuda:
            self.pwgan.cuda()
        self.pwgan.eval()

    def save_wav(self, wav, path):
        # wav *= 32767 / max(1e-8, np.max(np.abs(wav)))
        #wav = np.array(wav)
        self.ap.save_wav(wav, path)

    @staticmethod
    def split_into_sentences(text):
        text = " " + text + "  <stop>"
        text = text.replace("\n", " ")
        text = re.sub(prefixes, "\\1<prd>", text)
        text = re.sub(websites, "<prd>\\1", text)
        if "Ph.D" in text:
            text = text.replace("Ph.D.", "Ph<prd>D<prd>")
        text = re.sub(r"\s" + alphabets + "[.] ", " \\1<prd> ", text)
        text = re.sub(acronyms+" "+starters, "\\1<stop> \\2", text)
        text = re.sub(alphabets + "[.]" + alphabets + "[.]" + alphabets + "[.]", "\\1<prd>\\2<prd>\\3<prd>", text)
        text = re.sub(alphabets + "[.]" + alphabets + "[.]", "\\1<prd>\\2<prd>", text)
        text = re.sub(" "+suffixes+"[.] "+starters, " \\1<stop> \\2", text)
        text = re.sub(" "+suffixes+"[.]", " \\1<prd>", text)
        text = re.sub(" " + alphabets + "[.]", " \\1<prd>", text)
        if "”" in text:
            text = text.replace(".”", "”.")
        if "\"" in text:
            text = text.replace(".\"", "\".")
        if "!" in text:
            text = text.replace("!\"", "\"!")
        if "?" in text:
            text = text.replace("?\"", "\"?")
        text = text.replace(".", ".<stop>")
        text = text.replace("?", "?<stop>")
        text = text.replace("!", "!<stop>")
        text = text.replace("<prd>", ".")
        sentences = text.split("<stop>")
        sentences = sentences[:-1]
        sentences = list(filter(None, [s.strip() for s in sentences])) # remove empty sentences
        return sentences

    @staticmethod
    def parse_outputs(postnet_output, decoder_output, alignments):
        postnet_output = postnet_output[0].data.cpu().numpy()
        decoder_output = decoder_output[0].data.cpu().numpy()
        alignment = alignments[0].cpu().data.numpy()
        return postnet_output, decoder_output, alignment

    def vocode(self, postnet_outputs):
        postnet_output = np.vstack(postnet_outputs)

        if self.pwgan:
            vocoder_input = torch.FloatTensor(postnet_output.T).unsqueeze(0)
            if self.use_cuda:
                vocoder_input = vocoder_input.cuda()
            wav = self.pwgan.inference(vocoder_input, hop_size=self.ap.hop_length)
        elif self.wavernn:
            min, max = (-4., 4.) #(postnet_output.min(), postnet_output.max())
            vocoder_input = torch.FloatTensor(postnet_output.T).unsqueeze(0)
            if self.use_cuda:
                vocoder_input = vocoder_input.cuda()
            wav = self.wavernn.generate(torch.clamp((vocoder_input-min)/(max-min), 0, 1), batched=self.config.is_wavernn_batched, target=5500, overlap=550)
        else:
            wav = inv_spectrogram(postnet_output, self.ap, self.tts_config)

        return wav

    def tts(self, text):
        wavs = []
        sens = self.split_into_sentences(text)
        print(sens)

        postnet_outputs = []
        nframes = 0
        for sen in sens:
            # preprocess the given text
            inputs = text_to_seqvec(sen, self.tts_config, self.use_cuda)
            # synthesize voice
            decoder_output, postnet_output, alignments, _ = run_model(
                self.tts_model, inputs, self.tts_config, False, None, None)
            # convert outputs to numpy
            postnet_output, _, _ = self.parse_outputs(
                postnet_output, decoder_output, alignments)

            postnet_output = np.append(postnet_output, -4*np.ones([20, postnet_output.shape[1]], dtype=postnet_output.dtype), axis=0)
            postnet_outputs.append(postnet_output)
            nframes += postnet_output.shape[0]
            print("\t {} ".format(nframes))
            if nframes > 4000:  # to handle limited device memory
                nframes = 0
                wavs.append(self.vocode(postnet_outputs))
                postnet_outputs = []

        if len(postnet_outputs) > 0:
            wavs.append(self.vocode(postnet_outputs))

        out = io.BytesIO()
        #filter output to cleanup some synthesis artefacts
        w = sp.signal.sosfilt(self.sos, np.hstack(wavs))
        w = sp.signal.sosfilt(self.sos1, w)
        self.save_wav(w, out)

        return out
