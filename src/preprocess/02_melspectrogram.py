import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path

import librosa
import librosa.display

TRAIN_RESAMPLED_DIR = Path('../data/input/train_audio_resampled/')
TRAIN_MEL_DIR = Path('../data/input/train_audio_mel/')

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


def read_audio(conf, pathname, trim_long_data):
    y, sr = librosa.load(pathname, sr=conf.sampling_rate)
    if 0 < len(y):
        y, _ = librosa.effects.trim(y)

    if len(y) > conf.samples:
        if trim_long_data:
            y = y[0:0+conf.samples]
    else:
        padding = conf.samples - len(y)
        offset = padding // 2
        y = np.pad(y, (offset, conf.samples - len(y) - offset), 'constant')
    return y


def audio_to_melspectrogram(conf, audio):
    spectrogram = librosa.feature.melspectrogram(audio, 
                                                 sr=conf.sampling_rate,
                                                 n_mels=conf.n_mels,
                                                 hop_length=conf.hop_length,
                                                 n_fft=conf.n_fft,
                                                 fmin=conf.fmin,
                                                 fmax=conf.fmax)
    spectrogram = librosa.power_to_db(spectrogram)
    spectrogram = spectrogram.astype(np.float32)
    return spectrogram


def show_melspectrogram(conf, mels, title='Log-frequency power spectrogram'):
    librosa.display.specshow(mels, x_axis='time', y_axis='mel', 
                             sr=conf.sampling_rate, hop_length=conf.hop_length,
                            fmin=conf.fmin, fmax=conf.fmax)
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.show()


def read_as_melspectrogram(conf, pathname, trim_long_data, debug_display=False):
    x = read_audio(conf, pathname, trim_long_data)
    mels = audio_to_melspectrogram(conf, x)
    if debug_display:
        IPython.display.display(IPython.display.Audio(x, rate=conf.sampling_rate))
        show_melspectrogram(conf, mels)
    return mels


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


def main():
    train_df = pd.read_csv('../data/input/train.csv')
    train_audio_infos = train_df[['ebird_code', 'filename']].values.tolist()

    TRAIN_MEL_DIR.mkdir(parents=True, exist_ok=True)
    for ebird_code in train_df.ebird_code.unique():
        ebird_dir = TRAIN_MEL_DIR / ebird_code
        ebird_dir.mkdir(exist_ok=True)

    for ebird_code, file_name in tqdm(train_audio_infos):
        try:
            x = read_as_melspectrogram(conf, TRAIN_RESAMPLED_DIR / ebird_code / file_name.replace('.mp3', '.wav'), trim_long_data=False)
            x_color = mono_to_color(x)
            np.save(TRAIN_MEL_DIR / ebird_code / file_name.replace('.mp3', '.npy'), x_color)
        except Exception as e:
            with open(TRAIN_MEL_DIR / 'skipped.txt', 'a') as f:
                file_path = str(TRAIN_MEL_DIR / ebird_code / file_name)
                f.write(file_path + '\n')


if __name__ == '__main__':
    main()
