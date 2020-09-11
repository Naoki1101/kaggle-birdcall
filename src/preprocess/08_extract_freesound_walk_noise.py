import numpy as np
import pandas as pd
import librosa
import soundfile as sf


def main():
    df = pd.read_csv('../data/input/freesound/train_curated.csv')

    audio_length_list = []
    for idx in df.index:
        audio_fname = df.loc[idx, 'fname']
        audio, sr = librosa.load(f'../data/input/freesound/{audio_fname}', sr=44_100)
        audio_length = len(audio) // sr
        audio_length_list.append(audio_length)

    df['length'] = audio_length_list
    df = df[df['labels'] == 'Walk_and_footsteps'][df['length'] >= 3].reset_index(drop=True)

    all_noise = []
    for idx in df.index:
        fname = df.loc[idx, 'fname']
        noise, sr =  librosa.load(f'../data/input/freesound/{fname}', sr=44_100)
        all_noise.append(noise)

    all_noise = np.concatenate(all_noise)

    sf.write('../data/input/example_noise/freesound_walk_noise.wav', all_noise, samplerate=32_000)


if __name__ == '__main__':
    main()
