import numpy as np
import pandas as pd


def main():
    train_df = pd.read_csv('../data/input/train.csv')
    miss_idx = train_df[train_df['filename'] == 'XC195038.mp3'].index.values
    np.save('../pickle/miss_idx.npy', miss_idx)


if __name__ == '__main__':
    main()
