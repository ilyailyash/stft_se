import os
import io
import math
from argparse import ArgumentParser

import numpy as np
from scipy import fft
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pytorch_lightning as ptl
from pytorch_lightning import Trainer
import audiolib
from dataloader import LoadDataset
from DenoiserLoss import DenoiserLoss
from wavelets.Wavelet_DWT import Wavelet_DWT
from metrics import calc_si_sdr, calc_pesq, calc_stoi
from pytorch_lightning.logging import TestTubeLogger


class WaveletMask(nn.Module):
    def __init__(self,
                 window_length,
                 num_gru_levels,
                 wavelet_size,
                 wavelet_name=None,
                 hop_fraction=0.5,
                 hidden_size=512,
                 num_wavelet_layers=3):
        super(WaveletMask, self).__init__()
        # not the best model...
        self.window_length = window_length
        self.hidden_size = hidden_size
        self.num_wavelet_layers = num_wavelet_layers
        self.num_gru_levels = num_gru_levels
        self.wavelet_size = wavelet_size
        self.wavelet_name = wavelet_name
        self.wind = torch.FloatTensor(audiolib.hamming(window_length))

        self.hsize = int(hop_fraction * self.window_length)
        self.gru = nn.GRU(input_size=window_length, hidden_size=hidden_size, num_layers=num_gru_levels)
        self.fc = nn.Linear(in_features=hidden_size, out_features=window_length)
        self.wavelet = Wavelet_DWT(num_wavelet_layers=num_wavelet_layers, wavelet_size=wavelet_size, name=wavelet_name)

    def fraiming_padding(self, signal):
        ssize = signal.size(2)
        sstart = self.hsize - self.window_length
        send = ssize
        nframe = math.ceil((send - sstart) / self.hsize)
        zpleft = -sstart
        zpright = (nframe - 1) * self.hsize + self.window_length - zpleft - ssize

        if zpleft > 0 or zpright > 0:
            sigpad = torch.zeros((signal.size(0), signal.size(1), ssize + zpleft + zpright)).type_as(signal)
            sigpad[:, :, zpleft:sigpad.shape[2] - zpright] = signal
        else:
            sigpad = signal
        return sigpad, nframe, zpleft

    def framing(self, sigpad, nframe):
        frames = torch.zeros((sigpad.size(0), nframe, self.window_length)).type_as(sigpad)

        for i, frame_sampleindex in enumerate(range(0, nframe * self.hsize, self.hsize)):
            frames[:, i, :] = sigpad[:, 0, frame_sampleindex:frame_sampleindex +
                                     self.window_length]*self.wind.type_as(sigpad)
        return frames

    def unframing(self, x_enh, nframe, zpleft):
        sout = torch.zeros((x_enh.size(0), 1, nframe * self.hsize)).type_as(x_enh)
        x_old = torch.zeros(self.hsize).type_as(x_enh)
        for i, frame_sampleindex in enumerate(range(0, nframe * self.hsize, self.hsize)):
            sout[:, :, frame_sampleindex:frame_sampleindex +self.hsize] = x_old + x_enh[:, i:i + 1, 0:self.hsize]
            x_old = x_enh[:, i:i + 1, self.hsize:]
        sout = sout[:, :, zpleft:]
        return sout

    def decompose(self, signal):
        sigpad, nframe, zpleft = self.fraiming_padding(signal)
        frames = self.framing(sigpad, nframe)
        wavelet_decomposition = self.wavelet.decomposition(frames.reshape(-1, 1, self.window_length))
        wavelet_decomposition = wavelet_decomposition.reshape((-1, nframe, self.window_length))
        return wavelet_decomposition, nframe, zpleft

    def reconstruct(self, wavelet_decomposition, nframe, zpleft):
        wavelet_decomposition = wavelet_decomposition.reshape(-1, 1, self.window_length)
        reconstructed_signal = self.wavelet.reconstruction(wavelet_decomposition).reshape(-1, nframe,
                                                                                          self.window_length)
        sout = self.unframing(reconstructed_signal, nframe, zpleft)
        return sout

    def noise_denoising(self, noise, mask):
        noise_decomposition, nframe, zpleft = self.decompose(noise)
        masked_decomposition = noise_decomposition * mask
        sout = self.reconstruct(masked_decomposition, nframe, zpleft)
        return sout, masked_decomposition

    def forward(self, x, clean=None, noise=None):
        wavelet_decomposition, nframe, zpleft = self.decompose(x)
        self.gru.flatten_parameters()
        gru_output, _ = self.gru(wavelet_decomposition)
        mask = torch.sigmoid(self.fc(gru_output))
        masked_decomposition = wavelet_decomposition * mask
        sout = self.reconstruct(wavelet_decomposition, nframe, zpleft)
        if noise is not None:
            wavelet_clean, _, _ = self.decompose(clean)
            denoised_noise, noise_masked_decomposition = self.noise_denoising(noise, mask)
            return sout, denoised_noise, masked_decomposition, wavelet_clean, noise_masked_decomposition, mask
        else:
            return sout
