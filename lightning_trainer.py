import io

import numpy as np
from scipy import fft
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as ptl

from model import WaveletMask
from dataloader import LoadDataset
from DenoiserLoss import DenoiserLoss
from metrics import calc_si_sdr, calc_pesq, calc_stoi


class LightningWaveletMask(ptl.LightningModule):
    def __init__(self,
                 hparams):
        super(LightningWaveletMask, self).__init__()

        # not the best model...
        self.hparams = hparams

        self.loss = DenoiserLoss(alfa=hparams.alfa, beta=hparams.beta,
                                 wavelet_reg=hparams.wavelet_reg, net_reg=hparams.net_reg)
        self.train_loader = LoadDataset(ds_path=hparams.ds_path,
                                        train=True,
                                        normalize=True,
                                        sampling_rate=hparams.sampling_rate,
                                        val_part=hparams.val_part)

        self.val_loader = LoadDataset(ds_path=hparams.ds_path,
                                      train=False,
                                      normalize=True,
                                      sampling_rate=hparams.sampling_rate,
                                      val_part=hparams.val_part)

        self.model = WaveletMask(window_length=hparams.window_length,
                                 num_gru_levels=hparams.num_gru_levels,
                                 wavelet_size=hparams.wavelet_size,
                                 hop_fraction=hparams.hop_fraction,
                                 hidden_size=hparams.hidden_size,
                                 num_wavelet_layers=hparams.num_wavelet_layers,
                                 wavelet_name=hparams.wavelet_name)

    def draw_filters_fft(self):
        hi_f = np.abs(fft(self.model.wavelet.hi[0, 0, :].type_as(torch.zeros(0)).data.numpy()))
        lo_f = np.abs(fft(self.model.wavelet.lo[0, 0, :].type_as(torch.zeros(0)).data.numpy()))
        n = hi_f.shape[-1]
        m = np.max([hi_f.max(), lo_f.max()])
        plt.grid(True)
        plt.tight_layout()
        plt.axis([0, 1, 0, m + m * 0.05])
        plt.plot(np.arange(n // 2 + 1) / (n // 2), lo_f[:n // 2 + 1], 'k--', lw=2)
        plt.plot(np.arange(n // 2 + 1) / (n // 2), hi_f[:n // 2 + 1], 'k', lw=2)
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.clf()
        return buf

    def log_masks(self, mask, amount):

        mask = mask.cpu().numpy()
        fig, a = plt.subplots(1, amount, constrained_layout=True, figsize=(3*amount, 3))
        levels = MaxNLocator(nbins=15).tick_values(0, 1)
        cmap = plt.get_cmap('Spectral')
        norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
        x, y = np.arange(mask.shape[1]), np.arange(mask.shape[2])

        for i, a_i in enumerate(a):
            a_i.set_title('mask'+str(i))
            a_i.set_ylabel('Frame number')
            a_i.set_xlabel('Wavelet coefficient')
            im = a_i.pcolormesh(x, y, mask[i].transpose(1, 0), cmap=cmap, norm=norm)
            fig.colorbar(im, ax=a_i)

        self.logger.experiment.add_figure('mask', fig, global_step=self.global_step)

    def log_audio(self, noisy, denoised_signal, amount):
        for i in range(amount):
            self.logger.experiment.add_audio('noisy'+str(i), noisy[i], global_step=self.global_step,
                                             sample_rate=self.hparams.sampling_rate)
            self.logger.experiment.add_audio('denoised'+str(i), denoised_signal[i], global_step=self.global_step,
                                             sample_rate=self.hparams.sampling_rate)

    def forward(self, x, clean=None, noise=None):
        return self.model(x, clean, noise)

    def denoising_loss(self, denoised_signal, denoised_noise, to_loss_clean, clean, noise):
        return self.loss(denoised_signal, denoised_noise, to_loss_clean, clean, noise, self.model)

    def training_step(self, batch, batch_nb):
        noisy, clean, noise = batch
        denoised_signal, denoised_noise, masked_decomposition, \
            wavelet_clean, noise_masked_decomposition, mask = self.forward(noisy, clean, noise)
        loss, speech_loss, noise_loss, wavelet_loss = self.denoising_loss(masked_decomposition,
                                                                          noise_masked_decomposition,
                                                                          wavelet_clean, clean, noise)
        train_si_sdr = calc_si_sdr(clean, denoised_signal)
        progress_bar = {"full": loss,
                        "speech": speech_loss,
                        "noise": noise_loss}
        logs = {"train_loss": loss,
                "train_speech_loss": speech_loss,
                "train_noise_loss": noise_loss,
                "train_si_sdr": train_si_sdr}

        return {'loss': loss, 'progress_bar': progress_bar, 'log': logs}

    def validation_step(self, batch, batch_nb):
        noisy, clean, noise = batch
        denoised_signal, denoised_noise, masked_decomposition, \
            wavelet_clean, noise_masked_decomposition, mask = self.forward(noisy, clean, noise)
        loss, speech_loss, noise_loss, wavelet_loss = self.denoising_loss(masked_decomposition,
                                                                          noise_masked_decomposition,
                                                                          wavelet_clean, clean, noise)
        val_si_sdr = calc_si_sdr(clean, denoised_signal)
        logs = {"val_loss": loss,
                "val_speech_loss": speech_loss,
                "val_noise_loss": noise_loss,
                "val_wavelet_loss": wavelet_loss,
                "val_si_sdr": val_si_sdr}
        if batch_nb == 0:
            clean, denoised_signal = clean.squeeze().cpu(), denoised_signal.squeeze().cpu()

            pesq = calc_pesq(clean, denoised_signal, sample_rate=self.hparams.sampling_rate)
            stoi = calc_stoi(clean, denoised_signal, sample_rate=self.hparams.sampling_rate)
            self.logger.experiment.add_scalar('val_pesq', pesq, global_step=self.global_step)
            self.logger.experiment.add_scalar('val_stoi', stoi, global_step=self.global_step)

            self.log_masks(mask, 3)
            self.log_audio(clean, denoised_signal, 3)

        return logs

    def validation_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_speech_loss = torch.stack([x['val_speech_loss'] for x in outputs]).mean()
        avg_noise_loss = torch.stack([x['val_noise_loss'] for x in outputs]).mean()
        avg_wavelet_loss = torch.stack([x['val_wavelet_loss'] for x in outputs]).mean()
        avg_si_sdr = torch.stack([x['val_si_sdr'] for x in outputs]).mean()
        logs = {"val_loss": avg_loss,
                "val_speech_loss": avg_speech_loss,
                "val_noise_loss": avg_noise_loss,
                "val_wavelet_loss": avg_wavelet_loss,
                "val_si_sdr": avg_si_sdr}
        if self.hparams.wavelet_name is None:
            plot_buf = self.draw_filters_fft()
            image = np.array(Image.open(plot_buf))
            image = np.transpose(image, [2, 0, 1])
            self.logger.experiment.add_image("wavelet", image, global_step=self.global_step)
        return {'avg_val_loss': avg_loss, 'log': logs}

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, patience=1, verbose=True)
        return [optim], [scheduler]

    @ptl.data_loader
    def train_dataloader(self):
        return DataLoader(self.train_loader, batch_size=self.hparams.batch_size, shuffle=True,
                          pin_memory=False)

    @ptl.data_loader
    def val_dataloader(self):
        return DataLoader(self.val_loader, batch_size=self.hparams.batch_size, shuffle=True,
                          pin_memory=False)