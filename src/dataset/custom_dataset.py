import sys
import cv2
import random
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

sys.path.append('../src')
import factory


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
        filename = filename.replace('.mp3', '.npy')
        ebird_code = self.ebird_code[idx]
        image = np.load(f'../data/input/train_audio_mel/{ebird_code}/{filename}')

        if self.transforms:
            image = self.transforms(image=image)['image']

        image = cv2.resize(image, (self.cfg.img_size.height, self.cfg.img_size.width))
        image = image.transpose(2, 0, 1)
        image = (image / 255.0).astype(np.float32)

        label = self.labels[idx, :]
        return image, label
