# coding: utf-8
import torch
from torch import nn
from layers.tacotron import Encoder, Decoder, PostCBHG
from layers.gst_layers import GST
from utils.generic_utils import sequence_mask
from .gstnet import GSTNet

class TacotronGST(nn.Module):
    def __init__(self,
                 num_chars,
                 num_speakers,
                 r=5,
                 linear_dim=1025,
                 mel_dim=80,
                 memory_size=5,
                 attn_win=False,
                 attn_norm="sigmoid",
                 prenet_type="original",
                 prenet_dropout=True,
                 forward_attn=False,
                 trans_agent=False,
                 forward_attn_mask=False,
                 location_attn=True,
                 separate_stopnet=True,
                 text_gst=True):
        super(TacotronGST, self).__init__()
        self.r = r
        self.mel_dim = mel_dim
        self.linear_dim = linear_dim
        self.embedding = nn.Embedding(num_chars, 256)
        self.embedding.weight.data.normal_(0, 0.3)
        if num_speakers > 1:
            self.speaker_embedding = nn.Embedding(num_speakers, 256)
            self.speaker_embedding.weight.data.normal_(0, 0.3)
        self.encoder = Encoder(256)
        self.gst = GST(num_mel=80, num_heads=4, num_style_tokens=10, embedding_dim=32)
        self.decoder = Decoder(256+32, mel_dim, r, memory_size, attn_win,
                               attn_norm, prenet_type, prenet_dropout,
                               forward_attn, trans_agent, forward_attn_mask,
                               location_attn, separate_stopnet)
        self.postnet = PostCBHG(mel_dim)
        self.last_linear = nn.Linear(self.postnet.cbhg.gru_features * 2, linear_dim)

        if text_gst:
            self.textgst = GSTNet(self.gst.encoder.recurrence.input_size,
                                  self.gst.encoder.recurrence.hidden_size,
                                  32)
        else:
            self.textgst = None

    def forward(self, characters, text_lengths, mel_specs, speaker_ids=None):
        B = characters.size(0)
        mask = sequence_mask(text_lengths).to(characters.device)
        inputs = self.embedding(characters)
        encoder_outputs = self.encoder(inputs)
        encoder_outputs = self._add_speaker_embedding(encoder_outputs,
                                                      speaker_ids)

        textgst_outputs = self.textgst(encoder_outputs.detach(), speaker_ids=speaker_ids) if self.textgst else None #detach to prevent backprop through text-gst
        gst_outputs, _ = self.gst(mel_specs)
        gst_outputs = gst_outputs.expand(-1, encoder_outputs.size(1), -1)
        encoder_outputs = torch.cat((encoder_outputs, gst_outputs),2)
        mel_outputs, alignments, stop_tokens = self.decoder(
            encoder_outputs, mel_specs, mask)
        mel_outputs = mel_outputs.view(B, -1, self.mel_dim)
        linear_outputs = self.postnet(mel_outputs)
        linear_outputs = self.last_linear(linear_outputs)
        return mel_outputs, linear_outputs, alignments, stop_tokens, textgst_outputs

    def inference(self, characters, speaker_ids=None, style_mel=None, text_gst=True):
        B = characters.size(0)
        inputs = self.embedding(characters)
        encoder_outputs = self.encoder(inputs)
        encoder_outputs = self._add_speaker_embedding(encoder_outputs,
                                                      speaker_ids)

        if self.textgst is not None and text_gst:
            textgst_outputs = self.textgst(encoder_outputs.detach(), speaker_ids=speaker_ids)
            textgst_outputs = textgst_outputs.expand(encoder_outputs.size(0), encoder_outputs.size(1), -1)
            encoder_outputs = torch.cat((encoder_outputs, textgst_outputs), 2)
        elif style_mel is not None:
            gst_outputs = self.gst(style_mel)
            gst_outputs = gst_outputs.expand(-1, encoder_outputs.size(1), -1)
            encoder_outputs = torch.cat((encoder_outputs, gst_outputs), 2)
        else:
            nogst=torch.zeros([encoder_outputs.size(0), encoder_outputs.size(1), 32],dtype=encoder_outputs.dtype).to(encoder_outputs.device)
            encoder_outputs = torch.cat((encoder_outputs, nogst), 2)
            
        mel_outputs, alignments, stop_tokens = self.decoder.inference(
            encoder_outputs)
        mel_outputs = mel_outputs.view(B, -1, self.mel_dim)
        linear_outputs = self.postnet(mel_outputs)
        linear_outputs = self.last_linear(linear_outputs)
        return mel_outputs, linear_outputs, alignments, stop_tokens

    def _add_speaker_embedding(self, encoder_outputs, speaker_ids):
        if hasattr(self, "speaker_embedding") and speaker_ids is not None:
            speaker_embeddings = self.speaker_embedding(speaker_ids)

            speaker_embeddings.unsqueeze_(1)
            speaker_embeddings = speaker_embeddings.expand(encoder_outputs.size(0),
                                                           encoder_outputs.size(1),
                                                           -1)
            encoder_outputs = encoder_outputs + speaker_embeddings
        return encoder_outputs
