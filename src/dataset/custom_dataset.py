import sys
import random
import librosa
import numpy as np
import pandas as pd
import soundfile as sf
from torch.utils.data import Dataset

sys.path.append('../src')
import factory

PERIOD = 5


# https://www.kaggle.com/daisukelab/cnn-2d-basic-solution-powered-by-fast-ai
class conf:
    sampling_rate = 32_000
    duration = 2
    hop_length = 347*duration
    fmin = 20
    fmax = sampling_rate // 2
    n_mels = 256
    n_fft = n_mels * 20
    samples = sampling_rate * duration


def mono_to_color(
    X: np.ndarray, mean=None, std=None,
    norm_max=None, norm_min=None, eps=1e-6
):
    # Stack X as [X,X,X]
    X = np.stack([X, X, X], axis=-1)

    # Standardize
    mean = mean or X.mean()
    X = X - mean
    std = std or X.std()
    Xstd = X / (std + eps)
    _min, _max = Xstd.min(), Xstd.max()
    norm_max = norm_max or _max
    norm_min = norm_min or _min
    if (_max - _min) > eps:
        # Normalize to [0, 255]
        V = Xstd
        V[V < norm_min] = norm_min
        V[V > norm_max] = norm_max
        V = 255 * (V - norm_min) / (norm_max - norm_min)
        V = V.astype(np.uint8)
    else:
        # Just zero
        V = np.zeros_like(Xstd, dtype=np.uint8)
    return V


class CustomDataset(Dataset):
    def __init__(self, df, cfg):
        super().__init__()
        self.cfg = cfg
        self.filenames = df['filename'].values
        self.transforms = factory.get_transforms(self.cfg)
        if self.is_train:
            self.ebird_code = df['ebird_code'].values
            self.labels = df['target'].values
    
    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        ebird_code = self.ebird_code[idx]

        y, sr = sf.read(f'../data/input/train_audio/{ebird_code}/{filename}')

        len_y = len(y)
        effective_length = sr * PERIOD
        if len_y < effective_length:
            new_y = np.zeros(effective_length, dtype=y.dtype)
            start = np.random.randint(effective_length - len_y)
            new_y[start:start + len_y] = y
            y = new_y.astype(np.float32)
        elif len_y > effective_length:
            start = np.random.randint(len_y - effective_length)
            y = y[start:start + effective_length].astype(np.float32)
        else:
            y = y.astype(np.float32)

        melspec = librosa.feature.melspectrogram(y,
                                                 sr=conf.sampling_rate,
                                                 n_mels=conf.n_mels,
                                                 hop_length=conf.hop_length,
                                                 n_fft=conf.n_fft,
                                                 fmin=conf.fmin,
                                                 fmax=conf.fmax)
        melspec = librosa.power_to_db(melspec).astype(np.float32)

        image = mono_to_color(melspec)
        height, width, _ = image.shape
        image = cv2.resize(image, (int(width * self.img_size / height), self.img_size))
        image = np.moveaxis(image, 2, 0)
        image = (image / 255.0).astype(np.float32)

        if self.is_train:
            label = self.labels[idx]
            return image, label
        else:
            return image
