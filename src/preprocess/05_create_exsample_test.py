import numpy as np
import pandas as pd

import librosa

SR = 32_000


def main():
    example_df = pd.read_csv('../data/input/example_test_audio_metadata.csv')

    row_id_list = []
    audio_id_list = []
    site_list = []
    end_time_list = []
    target_list = []

    for fname in ['BLKFR-10-CPL_20190611_093000.pt540.mp3', 'ORANGE-7-CAP_20190606_093000.pt623.mp3']:
        clip, sr = librosa.load(f'../data/input/example_test_audio/{fname}',
                                            sr=SR,
                                            mono=True,
                                            res_type='kaiser_fast')
        len_clip = len(clip)
        df = example_df[example_df['file_id'] == fname.split('.')[0]].reset_index(drop=True)

        for i in range(len_clip // (SR * 5)):
            audio_id_list.append(fname)
            end_time_list.append((i + 1) * 5)
            row_id_list.append(f'{fname}_{(i + 1) * 5}')
            site_list.append('site_1')
            
            df_ = df[(df['time_start'] >= i * 5) & (df['time_end'] < (i + 1) * 5)]
            target_str = ' '.join(df_['ebird_code'].tolist())
            if len(target_str) == 0:
                target_str = 'nocall'
            target_list.append(target_str)
    
    example_target_df = pd.DataFrame({'row_id': row_id_list, 
                                      'site': site_list,
                                      'seconds': end_time_list, 
                                      'audio_id': audio_id_list, 
                                      'target': target_list})
    example_target_df = example_target_df.sort_values(by=['audio_id', 'seconds'])

    example_target_df.to_csv('../data/input/example_test.csv', index=False)


if __name__ == '__main__':
    main()
