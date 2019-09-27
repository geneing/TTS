import os
import argparse
import torch
import importlib

from utils.synthesis import synthesis
from utils.generic_utils import load_config, setup_model
from utils.text.symbols import symbols, phonemes
from utils.audio import AudioProcessor

from WaveRNN.models.wavernn import Model as VocoderModel

import spacy

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(54321)
use_cuda = torch.cuda.is_available()


def tts(model,
        vocoder_model,
        C,
        VC,
        text,
        ap,
        use_cuda,
        batched_vocoder=False,
        figures=False,
        text_gst=True):
    
    use_vocoder_model = vocoder_model is not None
    
    model.decoder.max_decoder_steps = 50000
    
    waveform, alignment, decoder_outputs, postnet_output, stop_tokens = synthesis(
        model, text=text, CONFIG=C, use_cuda=use_cuda, ap=ap, speaker_id=None, style_wav=None,
        enable_eos_bos_chars=C.enable_eos_bos_chars, text_gst=text_gst)
    if use_vocoder_model:
        vocoder_input = torch.FloatTensor(decoder_outputs.T).unsqueeze(0)
        waveform = vocoder_model.generate(
            vocoder_input.cuda() if use_cuda else vocoder_input,
            batched=batched_vocoder,
            target=11000,
            overlap=550)
    return alignment, postnet_output, stop_tokens, waveform

    
def setup_model(num_chars, num_speakers, c):
    print(" > Using model: {}".format(c.model))
    MyModel = importlib.import_module('models.' + c.model.lower())
    MyModel = getattr(MyModel, c.model)
    
    if c.model.lower() in ["tacotron"]:
        model = MyModel(
            num_chars=num_chars,
            num_speakers=num_speakers,
            r=c.r,
            linear_dim=1025,
            mel_dim=80,
            memory_size=c.memory_size,
            attn_win=c.windowing,
            attn_norm=c.attention_norm,
            prenet_type=c.prenet_type,
            prenet_dropout=c.prenet_dropout,
            forward_attn=c.use_forward_attn,
            trans_agent=c.transition_agent,
            forward_attn_mask=c.forward_attn_mask,
            location_attn=c.location_attn,
            separate_stopnet=c.separate_stopnet)
    elif c.model.lower() in ["tacotrongst"]:
        model = MyModel(
            num_chars=num_chars,
            num_speakers=num_speakers,
            r=c.r,
            linear_dim=1025,
            mel_dim=80,
            memory_size=c.memory_size,
            attn_win=c.windowing,
            attn_norm=c.attention_norm,
            prenet_type=c.prenet_type,
            prenet_dropout=c.prenet_dropout,
            forward_attn=c.use_forward_attn,
            trans_agent=c.transition_agent,
            forward_attn_mask=c.forward_attn_mask,
            location_attn=c.location_attn,
            separate_stopnet=c.separate_stopnet,
            text_gst=c.text_gst
        )

    elif c.model.lower() == "tacotron2":
        model = MyModel(
            num_chars=num_chars,
            num_speakers=num_speakers,
            r=c.r,
            attn_win=c.windowing,
            attn_norm=c.attention_norm,
            prenet_type=c.prenet_type,
            prenet_dropout=c.prenet_dropout,
            forward_attn=c.use_forward_attn,
            trans_agent=c.transition_agent,
            forward_attn_mask=c.forward_attn_mask,
            location_attn=c.location_attn,
            separate_stopnet=c.separate_stopnet)
    return model


if __name__ == "__main__":

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

    args = parser.parse_args()

    try:
        path = os.path.realpath(os.path.dirname(__file__))
    except NameError as e:
        path = './'
    
    
    C = load_config(os.path.join(path, 'pretrained_models/TTS/config.json'))
    C.forward_attn_mask = True

    # load the audio processor
    ap = AudioProcessor(**C.audio)
    num_speakers = 0

        # load the model
    num_chars = len(phonemes) if C.use_phonemes else len(symbols)
    model = setup_model(num_chars, num_speakers, C)
    cp = torch.load(os.path.join(path, 'pretrained_models/TTS/best_model.pth.tar'), map_location='cpu')
    model.load_state_dict(cp['model'], strict=False)
    model.r = cp['r']
    model.decoder.r = cp['r']
    model.eval()
    if use_cuda:
        model.cuda()

    VC = load_config(os.path.join(path, 'pretrained_models/WaveRNN/config.json'))
    bits = 10
    vocoder_model = VocoderModel(
        rnn_dims=512,
        fc_dims=512,
        mode=VC.mode,
        mulaw=VC.mulaw,
        pad=VC.pad,
        use_aux_net=VC.use_aux_net,
        use_upsample_net=VC.use_upsample_net,
        upsample_factors=VC.upsample_factors,
        feat_dims=VC.audio["num_mels"],
        compute_dims=128,
        res_out_dims=128,
        res_blocks=10,
        hop_length=ap.hop_length,
        sample_rate=ap.sample_rate,
    )

    check = torch.load(os.path.join(path, 'pretrained_models/WaveRNN/best_model.pth.tar'), map_location='cpu')
    vocoder_model.load_state_dict(check['model'])
    vocoder_model.eval()
    if use_cuda:
        vocoder_model.cuda()
    
    
    with open(args.input, 'r') as f:
        txt = f.read()
        nlp = spacy.load("en_core_web_sm")
        doc = nlp(txt)
        for i, sent in enumerate(doc.sents):
            print(sent.text, 
                  '\n\n')

            _, _, _, wav = tts(
                model,
                vocoder_model,
                C,
                VC,
                sent.text,
                ap,
                use_cuda,
                batched_vocoder=False,
                figures=False, text_gst=True)
            ap.save_wav(wav, '%s/%d.wav'%(args.output, i+1))
