import librosa
import audioread
import pandas as pd
import soundfile as sf
from pathlib import Path
from joblib import delayed, Parallel

import warnings
warnings.simplefilter('ignore')

TRAIN_AUDIO_DIR = Path('../data/input/train_audio/')
TRAIN_RESAMPLED_DIR = Path('../data/input/train_audio_resampled/')

TARGET_SR = 32_000
NUM_THREAD = 4  # for joblib.Parallel

# https://www.kaggle.com/c/birdsong-recognition/discussion/159943
def resample(ebird_code: str, filename: str, target_sr: int):    
    audio_dir = TRAIN_AUDIO_DIR
    resample_dir = TRAIN_RESAMPLED_DIR
    ebird_dir = resample_dir / ebird_code

    try:
        y, _ = librosa.load(
            str(audio_dir / ebird_code / filename),
            sr=target_sr, mono=True, res_type='kaiser_fast')

        filename = filename.replace('.mp3', '.wav')
        sf.write(ebird_dir / filename, y, samplerate=target_sr)
        return 'OK'
    except Exception as e:
        with open(resample_dir / 'skipped.txt', 'a') as f:
            file_path = str(audio_dir / ebird_code / filename)
            f.write(file_path + '\n')
        return str(e)


def main():
    train_df = pd.read_csv('../data/input/train.csv')
    train_audio_infos = train_df[['ebird_code', 'filename']].values.tolist()

    TRAIN_RESAMPLED_DIR.mkdir(parents=True, exist_ok=True)
    for ebird_code in train_df.ebird_code.unique():
        ebird_dir = TRAIN_RESAMPLED_DIR / ebird_code
        ebird_dir.mkdir(exist_ok=True)

    msg_list = Parallel(n_jobs=NUM_THREAD, verbose=1)(
        delayed(resample)(ebird_code, file_name, TARGET_SR) for ebird_code, file_name in train_audio_infos)

    train_df['resampled_sampling_rate'] = TARGET_SR
    train_df['resampled_filename'] = train_df['filename'].map(
        lambda x: x.replace('.mp3', '.wav'))
    train_df['resampled_channels'] = '1 (mono)'

    train_df[['ebird_code', 'filename', 'resampled_sampling_rate', 'resampled_filename', 'resampled_channels']].to_csv(
        '../data/input/train_mod.csv', index=False
    )


if __name__ == '__main__':
    main()
