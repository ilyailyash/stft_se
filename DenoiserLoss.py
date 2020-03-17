import torch
import torch.nn as nn
import torch.nn.functional as F



EPS = torch.finfo(float).eps


def rmse(clean, noisy):
    criterion = nn.MSELoss()
    loss = torch.sqrt(criterion(clean, noisy))
    return loss


def signal_noise_ratio(clean, noise):
    snr = 10*torch.log10(clean.pow(2).sum() / noise.pow(2).sum())
    return snr





class WaveletLoss(nn.Module):
    def __init__(self):
        super(WaveletLoss, self).__init__()

    @staticmethod
    def orthonormal(w_1, w_2=None):
        kernel_size = w_1.size(2)
        if w_2 is None:
            w_2 = w_1
            start = 1
            wavelet_restriction = (w_1.pow(2).sum()-1).pow(2)
        else:
            start = 0
            wavelet_restriction = torch.zeros(1).type_as(w_1)

        for m in range(start, w_1.size(2) // 2):
            tmp = torch.zeros((1, 1, 1)).type_as(w_1)
            prods = [w_1[:, :, i] * w_2[:, :, i + 2 * m] for i in range(kernel_size - 2 * m)]
            for n in prods:
                tmp += n
            wavelet_restriction += tmp[0, 0, 0].pow(2)
        return wavelet_restriction

    def forward(self, wavelet):
        w_hi = wavelet.hi
        w_lo = wavelet.lo

        wavelet_restriction1 = self.orthonormal(w_hi)
        wavelet_restriction2 = self.orthonormal(w_lo)
        wavelet_restriction3 = w_hi.sum().pow(2)
        wavelet_restriction4 = (w_lo.sum() - 2 ** (1 / 2)).pow(2)
        wavelet_restriction5 = self.orthonormal(w_lo, w_hi)

        return (wavelet_restriction1 + wavelet_restriction2 + wavelet_restriction3 +
                wavelet_restriction4 + wavelet_restriction5)


class DenoiserLoss(nn.Module):
    def __init__(self, wavelet_reg, alfa=None, beta=None,  net_reg=None):
        super(DenoiserLoss, self).__init__()

        self.alfa = alfa
        self.beta = beta
        assert alfa is not None or beta is not None, 'alfa of beta must be defined'
        self.wavelet_reg = wavelet_reg
        self.net_reg = net_reg

        self.criterion = nn.MSELoss()
        self.wavelet_criterion = WaveletLoss()

        self.average_filter = nn.Parameter(torch.tensor([[[1 / 3, 1 / 3, 1 / 3]]]), requires_grad=False)

    def voice_detection(self, clean):
        frame_rms = 10 * torch.log10(torch.sum(torch.pow(clean, 2) + EPS, dim=2))
        threshold = frame_rms.max() - 30
        frame_rms = F.conv1d(frame_rms.unsqueeze(1), self.average_filter, padding=self.average_filter.size(-1) // 2)
        mask = frame_rms.squeeze(1) > threshold
        return mask

    def get_alfa(self, snr):
        return snr/(snr+self.beta)

    def get_speech_loss(self, denoised_signal, clean):
        mask = self.voice_detection(clean)
        return self.criterion(denoised_signal[mask], clean[mask])

    @staticmethod
    def get_noise_loss(denoised_noise):
        return denoised_noise.pow(2).mean()

    @staticmethod
    def signal_noise_ratio(clean, noise):
        snr = 10 * torch.log10(clean.pow(2).sum() / noise.pow(2).sum())
        return snr

    def forward(self, denoised_signal, denoised_noise, to_loss_clean, clean, noise, net):

        wavelet_loss = self.wavelet_criterion(net.wavelet)
        speech_loss = self.get_speech_loss(denoised_signal, to_loss_clean)
        noise_loss = self.get_noise_loss(denoised_noise)

        if self.alfa is None:
            self.alfa = self.get_alfa(self.signal_noise_ratio(clean, noise))
        data_loss = self.alfa*speech_loss + (1-self.alfa)*noise_loss

        regularization = self.wavelet_reg * wavelet_loss
        if self.net_reg is not None:
            l2_reg = torch.tensor(0.).type_as(denoised_signal)
            for name, param in net.named_parameters():
                if param.requires_grad:
                    if 'wavelet' in name:
                        continue
                    else:
                        l2_reg += torch.norm(param)
            regularization += self.net_reg * l2_reg

        return data_loss + regularization, speech_loss, noise_loss, wavelet_loss


if __name__ == "__main__":
    from model import WaveletMask
    net = WaveletMask(512, 3, wavelet_size=16, wavelet_name='db8')
    criterion = DenoiserLoss(wavelet_reg=0.03, alfa=0.35, beta=None, net_reg=1e-6)
    a = torch.tensor([[[1., 2., 3., 4., 5.]],
                      [[1., 2., 3., 4., 5.]]])
    b = torch.zeros_like(a)

    print (criterion(b, b, b, b, net))
