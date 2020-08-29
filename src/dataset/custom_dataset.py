import sys
import cv2
import random
import librosa
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

sys.path.append('../src')
import factory


class conf:
    duration = 5
    sampling_rate = 32_000
    n_mels = 128
    fmin = 20
    fmax = sampling_rate // 2
    samples = sampling_rate * duration


def mono_to_color(X, mean=None, std=None, norm_max=None, norm_min=None, eps=1e-6):
    # Stack X as [X,X,X]
    X = np.stack([X, X, X], axis=-1)

    # Standardize
    mean = mean or X.mean()
    std = std or X.std()
    Xstd = (X - mean) / (std + eps)
    _min, _max = Xstd.min(), Xstd.max()
    norm_max = norm_max or _max
    norm_min = norm_min or _min
    if (_max - _min) > eps:
        # Scale to [0, 255]
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
    def __init__(self, df, target_df, cfg):
        super().__init__()
        self.cfg = cfg
        self.filenames = df['filename'].values
        self.labels = target_df.values.astype(float)
        self.transforms = factory.get_transforms(self.cfg)
        self.is_train = cfg.is_train
        self.ebird_code = df['ebird_code'].values

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        filename = filename.replace('.mp3', '.wav')
        ebird_code = self.ebird_code[idx]
        path_name = f'../data/input/train_audio_resampled/{ebird_code}/{filename}'

        y, sr = librosa.load(path_name, sr=conf.sampling_rate)

        len_y = len(y)
        if len_y < conf.samples:
            padding = conf.samples - len_y
            offset = padding // 2
            y = np.pad(y, (offset, conf.samples - len_y - offset), 'constant').astype(np.float32)
        elif len_y > conf.samples:
            start = np.random.randint(len_y - conf.samples)
            y = y[start: start + conf.samples].astype(np.float32)
        else:
            y = y.astype(np.float32)

        melspec = librosa.feature.melspectrogram(y,
                                                 sr=conf.sampling_rate,
                                                 n_mels=conf.n_mels,
                                                 fmin=conf.fmin,
                                                 fmax=conf.fmax)
        melspec = librosa.power_to_db(melspec).astype(np.float32)

        image = mono_to_color(melspec)

        if self.transforms:
            image = self.transforms(image=image)['image']

        image = cv2.resize(image, (self.cfg.img_size.height, self.cfg.img_size.width))
        image = image.transpose(2, 0, 1)
        image = (image / 255.0).astype(np.float32)

        label = self.labels[idx, :]
        return image, label
