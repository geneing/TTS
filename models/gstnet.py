# coding: utf-8
import torch
from torch import nn
from layers.tacotron import Encoder, Decoder, PostCBHG
from layers.gst_layers import GST


class GSTNet(nn.Module):
    def __init__(self, input_size=256, hidden_size=128, style_dim=128):
        super(GSTNet, self).__init__()
        self.aggregating_gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, style_dim)
        self._gru_memory = None

    def forward(self, text_encoder_input, speaker_ids=None, persistent=False):
        self.aggregating_gru.flatten_parameters()
        if persistent:
            _, self._gru_memory = self.aggregating_gru(text_encoder_input, self._gru_memory)
        else:
            _, self._gru_memory = self.aggregating_gru(text_encoder_input)

        fc_out = self.fc(self._gru_memory.squeeze(0))
        output = torch.tanh(fc_out)
        return output
