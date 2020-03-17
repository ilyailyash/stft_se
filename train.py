from argparse import ArgumentParser
from pytorch_lightning import Trainer
from pytorch_lightning.logging import TestTubeLogger
from lightning_trainer import LightningWaveletMask

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--ds_path', type=str, default='/home/administrator/Data/DNS-Challenge/training')
    parser.add_argument('--sampling_rate', type=int, default=16000)
    parser.add_argument('--window_length', type=int, default=512)
    parser.add_argument('--hop_fraction', type=float, default=0.5)
    parser.add_argument('--val_part', type=float, default=0.05)
    parser.add_argument('--num_gru_levels', type=int, default=3)
    parser.add_argument('--num_wavelet_layers', type=int, default=5)
    parser.add_argument('--wavelet_size', type=int, default=16)
    parser.add_argument('--wavelet_name', type=str, default='db8')
    parser.add_argument('--lr', type=float, default=6e-2)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--net_reg', type=float, default=1e-6)
    parser.add_argument('--wavelet_reg', type=float, default=0.0)
    parser.add_argument('--alfa', type=float, default=0.35)
    parser.add_argument('--beta', type=float, default=20)
    parser.add_argument('--hidden_size', type=float, default=512)

    hparams = parser.parse_args()

    model = LightningWaveletMask(hparams)

    logger = TestTubeLogger(
        save_dir='./lightning_logs',
        version=0  # An existing version with a saved checkpoint
    )
    trainer = Trainer(gpus=[0, 1],
                      # num_sanity_val_steps=0,
                      # amp_level='O3',
                      # distributed_backend='dp',
                      # use_amp=True,
                      # val_check_interval=1,
                      show_progress_bar=True)

    trainer.fit(model)