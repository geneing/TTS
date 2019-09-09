# coding: utf-8
from torch import nn
from layers.tacotron import Encoder, Decoder, PostCBHG
from layers.gst_layers import GST
from utils.generic_utils import sequence_mask


class GSTNet(nn.Module):
    def __init__(self, input_size = 256, hidden_size=128, ntokens = 10 ):
        super(GSTNet, self).__init__()
        self.rnn = nn.GRU(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, ntokens)

    def forward(self):
        pass