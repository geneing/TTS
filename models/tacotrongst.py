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
                 postnet_output_dim=1025,
                 decoder_output_dim=80,
                 memory_size=5,
                 attn_type='original',
                 attn_win=False,
                 gst=False,
                 attn_norm="sigmoid",
                 prenet_type="original",
                 prenet_dropout=True,
                 forward_attn=False,
                 trans_agent=False,
                 forward_attn_mask=False,
                 location_attn=True,
                 attn_K=5,
                 separate_stopnet=True,
                 bidirectional_decoder=False,
                 text_gst=True):
        super(TacotronGST, self).__init__()
        self.r = r
        self.gst = gst
        self.mel_dim = decoder_output_dim
        self.linear_dim = postnet_output_dim
        self.bidirectional_decoder = bidirectional_decoder
        decoder_dim = 512 if num_speakers > 1 else 256
        encoder_dim = 512 if num_speakers > 1 else 256
        proj_speaker_dim = 80 if num_speakers > 1 else 0
        # embedding layer
        self.embedding = nn.Embedding(num_chars, 256)
        self.embedding.weight.data.normal_(0, 0.3)
        if num_speakers > 1:
            self.speaker_embedding = nn.Embedding(num_speakers, 256)
            self.speaker_embedding.weight.data.normal_(0, 0.3)
            
        self.encoder = Encoder(encoder_dim)
        self.decoder = Decoder(decoder_dim+32, self.mel_dim, r, memory_size, attn_type, attn_win,
                               attn_norm, prenet_type, prenet_dropout,
                               forward_attn, trans_agent, forward_attn_mask,
                               location_attn, attn_K, separate_stopnet,
                               proj_speaker_dim)
        if self.bidirectional_decoder:
            self.decoder_backward = copy.deepcopy(self.decoder)

        self.postnet = PostCBHG(self.mel_dim)
        self.last_linear = nn.Linear(self.postnet.cbhg.gru_features * 2, self.linear_dim)
        
        # speaker embedding layers
        self.num_speakers = num_speakers
        self.speaker_embeddings_projected = None
        if num_speakers > 1:
            self.speaker_embedding = nn.Embedding(num_speakers, 256)
            self.speaker_embedding.weight.data.normal_(0, 0.3)
            self.speaker_project_mel = nn.Sequential(
                nn.Linear(256, proj_speaker_dim), nn.Tanh())
            self.speaker_embeddings = None
        # global style token layers
        gst_embedding_dim = 32
        if self.gst or text_gst:
            self.gst_layer = GST(num_mel=80,
                                 num_heads=4,
                                 num_style_tokens=10,
                                 embedding_dim=gst_embedding_dim)
            
        if text_gst:
            self.textgst = GSTNet(self.gst_layer.encoder.recurrence.input_size,
                                  self.gst_layer.encoder.recurrence.hidden_size,
                                  gst_embedding_dim)
        else:
            self.textgst = None

    def _init_states(self):
        self.speaker_embeddings = None
        self.speaker_embeddings_projected = None

    def compute_speaker_embedding(self, speaker_ids):
        if hasattr(self, "speaker_embedding") and speaker_ids is None:
            raise RuntimeError(
                " [!] Model has speaker embedding layer but speaker_id is not provided"
            )
        if hasattr(self, "speaker_embedding") and speaker_ids is not None:
            self.speaker_embeddings = self._compute_speaker_embedding(
                speaker_ids)
            self.speaker_embeddings_projected = self.speaker_project_mel(
                self.speaker_embeddings).squeeze(1)

    def compute_gst(self, inputs, mel_specs):
        gst_outputs, _ = self.gst_layer(mel_specs)
        inputs = self._add_speaker_embedding(inputs, gst_outputs)
        return inputs

    def forward(self, characters, text_lengths, mel_specs, speaker_ids=None):
        """
        Shapes:
            - characters: B x T_in
            - text_lengths: B
            - mel_specs: B x T_out x D
            - speaker_ids: B x 1
        """
        self._init_states()
        mask = sequence_mask(text_lengths).to(characters.device)
        inputs = self.embedding(characters)
        # B x speaker_embed_dim
        self.compute_speaker_embedding(speaker_ids)
        if self.num_speakers > 1:
            # B x T_in x embed_dim + speaker_embed_dim
            inputs = self._concat_speaker_embedding(inputs,
                                                    self.speaker_embeddings)
        
        encoder_outputs = self.encoder(inputs)
        if self.gst:
            # B x gst_dim
            encoder_outputs = self.compute_gst(encoder_outputs, mel_specs)
        if self.num_speakers > 1:
            encoder_outputs = self._concat_speaker_embedding(
                encoder_outputs, self.speaker_embeddings)

        textgst_outputs = self.textgst(encoder_outputs.detach(), speaker_ids=speaker_ids) if self.textgst else None #detach to prevent backprop through text-gst

        gst_outputs, _ = self.gst_layer(mel_specs)
        gst_outputs = gst_outputs.expand(encoder_outputs.size(0), encoder_outputs.size(1), -1)
        encoder_outputs = torch.cat((encoder_outputs, gst_outputs),dim=-1)
        
        # decoder_outputs: B x decoder_dim x T_out
        # alignments: B x T_in x encoder_dim
        # stop_tokens: B x T_in
        mel_outputs, alignments, stop_tokens = self.decoder(
            encoder_outputs, mel_specs, mask, self.speaker_embeddings_projected)
        # B x T_out x decoder_dim
        linear_outputs = self.postnet(mel_outputs)
        linear_outputs = self.last_linear(linear_outputs)

        # B x T_out x decoder_dim
        mel_outputs = mel_outputs.transpose(1, 2).contiguous()
        if self.bidirectional_decoder:
            mel_outputs_backward, alignments_backward = self._backward_inference(mel_specs, encoder_outputs, mask)
        else:
            mel_outputs_backward = None
            alignments_backward = None
        
        return mel_outputs, linear_outputs, alignments, stop_tokens, mel_outputs_backward, alignments_backward, textgst_outputs

    def inference(self, characters, speaker_ids=None, style_mel=None, text_gst=True):
        B = characters.size(0)
        inputs = self.embedding(characters)
        self._init_states()
        self.compute_speaker_embedding(speaker_ids)
        if self.num_speakers > 1:
            inputs = self._concat_speaker_embedding(inputs,
                                                    self.speaker_embeddings)
        
        encoder_outputs = self.encoder(inputs)
        if self.gst and style_mel is not None:
            encoder_outputs = self.compute_gst(encoder_outputs, style_mel)
        if self.num_speakers > 1:
            encoder_outputs = self._concat_speaker_embedding(
                encoder_outputs, self.speaker_embeddings)

        if self.textgst is not None and text_gst:
            textgst_outputs = self.textgst(encoder_outputs.detach(), speaker_ids=speaker_ids)
            textgst_outputs = textgst_outputs.expand(encoder_outputs.size(0), encoder_outputs.size(1), -1)
            encoder_outputs = torch.cat((encoder_outputs, textgst_outputs), 2)
        elif style_mel is not None:
            gst_outputs = self.gst_layer(style_mel)
            gst_outputs = gst_outputs.expand(-1, encoder_outputs.size(1), -1)
            encoder_outputs = torch.cat((encoder_outputs, gst_outputs), 2)
        else:
            nogst=torch.zeros([encoder_outputs.size(0), encoder_outputs.size(1), 32],dtype=encoder_outputs.dtype).to(encoder_outputs.device)
            encoder_outputs = torch.cat((encoder_outputs, nogst), 2)
            
        mel_outputs, alignments, stop_tokens = self.decoder.inference(
            encoder_outputs)
        linear_outputs = self.postnet(mel_outputs)
        linear_outputs = self.last_linear(linear_outputs)
        mel_outputs = mel_outputs.transpose(1,2).contiguous()
        return mel_outputs, linear_outputs, alignments, stop_tokens

    def _backward_inference(self, mel_specs, encoder_outputs, mask):
        decoder_outputs_b, alignments_b, _ = self.decoder_backward(
            encoder_outputs, torch.flip(mel_specs, dims=(1,)), mask,
            self.speaker_embeddings_projected)
        decoder_outputs_b = decoder_outputs_b.transpose(1, 2).contiguous()
        return decoder_outputs_b, alignments_b

    def _compute_speaker_embedding(self, speaker_ids):
        speaker_embeddings = self.speaker_embedding(speaker_ids)
        return speaker_embeddings.unsqueeze_(1)

    @staticmethod
    def _add_speaker_embedding(outputs, speaker_embeddings):
        speaker_embeddings_ = speaker_embeddings.expand(
            outputs.size(0), outputs.size(1), -1)
        outputs = outputs + speaker_embeddings_
        return outputs

    @staticmethod
    def _concat_speaker_embedding(outputs, speaker_embeddings):
        speaker_embeddings_ = speaker_embeddings.expand(
            outputs.size(0), outputs.size(1), -1)
        outputs = torch.cat([outputs, speaker_embeddings_], dim=-1)
        return outputs
