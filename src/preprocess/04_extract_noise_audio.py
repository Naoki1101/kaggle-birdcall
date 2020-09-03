import numpy as np
import pandas as pd
import soundfile as sf
from pathlib import Path

import librosa

SR = 32_000
DATE_DIR = Path('../data/input/example_test_audio')
ORANGE_FNAME = 'ORANGE-7-CAP_20190606_093000.pt623.mp3'
BLKFR_FNAME = 'BLKFR-10-CPL_20190611_093000.pt540.mp3'


def extract_noise(df, audio_path):
    df['pre_time_end'] = df['time_end'].shift(1)
    df['noise_time'] = df['time_start'] - df['pre_time_end']
    df_noise = df[df['noise_time'] >= 3].reset_index(drop=True)

    sample, sr = librosa.load(audio_path,
                              sr=SR,
                              mono=True,
                              res_type='kaiser_fast')

    noise_list = []
    for idx in range(len(df_noise)):
        e = df_noise.loc[idx, 'time_start'] * SR
        s = df_noise.loc[idx, 'pre_time_end'] * SR
        noise_list.append(sample[int(s): int(e)])

    return noise_list



def main():
    df = pd.read_csv('../data/input/example_test_audio_metadata.csv', 
                     usecols=['file_id', 'ebird_code', 'time_start', 'time_end'])

    df_orange = df[df['file_id'] == 'ORANGE-7-CAP_20190606_093000'].reset_index(drop=True)
    df_blkfr = df[df['file_id'] == 'BLKFR-10-CPL_20190611_093000'].reset_index(drop=True)

    orange_noise_list = extract_noise(df_orange, str(DATE_DIR / ORANGE_FNAME))
    blkfr_noise_list = extract_noise(df_blkfr, str(DATE_DIR / BLKFR_FNAME))
    
    noise = np.concatenate([
        np.concatenate(orange_noise_list),
        np.concatenate(blkfr_noise_list)
    ])

    sf.write('../data/input/example_noise/example_noise.wav', noise, samplerate=SR)


if __name__ == '__main__':
    main()
