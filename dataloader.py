import os
import glob
import re
import time
import random
from random import shuffle

import scipy.io.wavfile as sci_wav
import librosa
import torch
from torch.utils.data import Dataset
import numpy as np


class LoadDataset(Dataset):
    def __init__(self,
                 ds_path,
                 train,
                 sampling_rate,
                 normalize=True,
                 audioformat='*wav',
                 val_part=0.05):

        self.ds_path = ds_path
        self.noisyspeechdir = os.path.join(ds_path, 'noisy')
        self.cleanspeechdir = os.path.join(ds_path, 'clean')
        self.noisedir = os.path.join(ds_path, 'noise')
        self.train = train
        self.sampling_rate = sampling_rate
        self.audioformat = audioformat
        self.normalize = normalize

        self.val_part = val_part

        random.seed(3)

        input_filelist = glob.glob(os.path.join(self.noisyspeechdir, self.audioformat))
        shuffle(input_filelist)

        n_val_part = int(len(input_filelist) * self.val_part)
        if self.train:
            self.files = input_filelist[:-n_val_part]
        else:
            self.files = input_filelist[-n_val_part:]

        random.seed(time.time())


    def __getitem__(self, index):
        return self.pull_item(index)

    def __len__(self):
        return len(self.files)

    def _to_tensor(self, signal):
        return torch.tensor(signal, dtype=torch.float32)

    def _norm(self, noisy, clean, noise):
        scale = 2**15 - 1
        return (noisy / scale, clean / scale, noise / scale)

    def fname_to_wav(self, fname):
        """Retrive wav data from fname
        """
        fileid = re.search('_fileid_\d*', fname)[0]
        clean_fname = os.path.join(self.cleanspeechdir, 'clean' + fileid + '.wav')
        noise_fname = os.path.join(self.noisedir, 'noise' + fileid + '.wav')

        sr, noisy = sci_wav.read(fname)
        if sr != self.sampling_rate: librosa.resample(noisy, sr, self.sampling_rate)
        sr, clean = sci_wav.read(clean_fname)
        if sr != self.sampling_rate: librosa.resample(clean, sr, self.sampling_rate)
        sr, noise = sci_wav.read(noise_fname)
        if sr != self.sampling_rate: librosa.resample(noise, sr, self.sampling_rate)

        return noisy, clean, noise

    def pull_item(self, index):
        noisy, clean, noise = self.fname_to_wav(self.files[index])
        if self.normalize:
            noisy, clean, noise = self._norm(noisy, clean, noise)
        return (self._to_tensor(noisy).unsqueeze(0), self._to_tensor(clean).unsqueeze(0),
                self._to_tensor(noise).unsqueeze(0))


if __name__ == "__main__":
    import math
    loader = LoadDataset(ds_path='/home/administrator/Data/DNS-Challenge/training',
                         train=True, normalize=True,
                         sampling_rate=16000, val_part=0.05)

    dataloader = torch.utils.data.DataLoader(loader, 100,
                                             shuffle=True,
                                             pin_memory=False)

    noisy, clean, noise = next(iter(dataloader))

    ssize = noisy.size(2)
    print('ssize:', ssize)
    framesize = int(0.032 * 16000)
    hopsize = int(framesize * 0.5)
    fsize = framesize
    hsize = hopsize



    sstart = hsize - fsize
    print('sstart:', sstart)
    send = ssize
    nframe = math.ceil((send - sstart) / hsize)
    zpleft = -sstart
    zpright = (nframe - 1) * hsize + fsize - zpleft - ssize

    if zpleft > 0 or zpright > 0:
        sigpad = torch.zeros((noisy.size(0), noisy.size(1), ssize + zpleft + zpright))
        print(ssize + zpleft + zpright)
        sigpad[:, :, zpleft:sigpad.shape[2] - zpright] = noisy
    else:
        sigpad = noisy

    print(sigpad.shape, noisy.shape)
    print(noisy.shape, clean.shape, noise.shape, len(dataloader))
    print(clean.max(), noise.min())