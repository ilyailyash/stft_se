import numpy as np
import torch
from pypesq import pesq
from pystoi.stoi import stoi


def calc_metric(reference, estimation, sample_rate, metric):
    if reference.ndim >= 2:
        return np.mean([
            metric(x_entry, y_entry, sample_rate)
            for x_entry, y_entry in zip(reference, estimation)
        ])
    else:
        return metric(reference, estimation, sample_rate)

def calc_pesq(reference, estimation, sample_rate):
    return calc_metric(reference.squeeze().cpu().data, estimation.squeeze().cpu().data, sample_rate, pesq)


def calc_stoi(reference, estimation, sample_rate):
    return calc_metric(reference.squeeze().cpu().data, estimation.squeeze().cpu().data, sample_rate, stoi)

def calc_si_sdr(reference, estimation):
    return torch.mean(si_sdr(reference.squeeze(), estimation.squeeze()))


"""
@InProceedings{Drude2017DeepClusteringIntegration,
  Title                    = {Tight integration of spatial and spectral features for {BSS} with Deep Clustering embeddings},
  Author                   = {Drude, Lukas and and Haeb-Umbach, Reinhold},
  Booktitle                = {INTERSPEECH 2017, Stockholm, Sweden},
  Year                     = {2017},
  Month                    = {Aug}
}
"""
def si_sdr(reference, estimation, *args):
    """
    Scale-Invariant Signal-to-Distortion Ratio (SI-SDR)
    Args:
        reference: numpy.ndarray, [..., T]
        estimation: numpy.ndarray, [..., T]
    Returns:
        SI-SDR
    [1] SDR– Half- Baked or Well Done?
    http://www.merl.com/publications/docs/TR2019-013.pdf
    >>> np.random.seed(0)
    >>> reference = torch.tensor(np.random.randn(100))
    >>> si_sdr(reference, reference)
    inf
    >>> si_sdr(reference, reference * 2)
    inf
    >>> si_sdr(reference, np.flip(reference))
    -25.127672346460717
    >>> si_sdr(reference, reference + np.flip(reference))
    0.481070445785553
    >>> si_sdr(reference, reference + 0.5)
    6.3704606032577304
    >>> si_sdr(reference, reference * 2 + 1)
    6.3704606032577304
    >>> si_sdr([1., 0], [0., 0])  # never predict only zeros
    nan
    >>> si_sdr([reference, reference], [reference * 2 + 1, reference * 1 + 0.5])
    array([6.3704606, 6.3704606])
    """
    estimation, reference = torch.broadcast_tensors(estimation, reference)

    # assert reference.dtype == np.float64, reference.dtype
    # assert estimation.dtype == np.float64, estimation.dtype

    reference_energy = torch.sum(reference ** 2, dim=-1, keepdim=True)

    # This is $\alpha$ after Equation (3) in [1].
    optimal_scaling = torch.sum(reference * estimation, dim=-1, keepdim=True) \
                      / reference_energy

    # This is $e_{\text{target}}$ in Equation (4) in [1].
    projection = optimal_scaling * reference

    # This is $e_{\text{res}}$ in Equation (4) in [1].
    noise = estimation - projection

    ratio = torch.sum(projection ** 2, dim=-1) / torch.sum(noise ** 2, dim =-1)
    return 10 * torch.log10(ratio)

if __name__ == '__main__':
    reference, estimation, sample_rate = torch.rand([5,1,480000]).cuda(), torch.rand([5,1,480000]).cuda(), 16000

    pesq_val = calc_pesq(reference, estimation, sample_rate)
    stoi_val = calc_stoi(reference, estimation, sample_rate)
    si_sdr_val = calc_si_sdr(reference, estimation, sample_rate)
    print(pesq_val, stoi_val, si_sdr_val)
