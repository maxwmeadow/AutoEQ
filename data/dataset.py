import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

SPECTROGRAM_DIRECTORY = 'data/processed/spectrograms'
LABELS_FILE = 'data/processed/labels.csv'

class AutoEQDataset(Dataset):
    def __init__(self, data_directory='data/processed'):
        self.spectrogram_directory = os.path.join(data_directory, 'spectrograms')
        self.labels = pd.read_csv(os.path.join(data_directory, 'labels.csv'))
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        row = self.labels.iloc[index]

        spectrogram = np.load(os.path.join(self.spectrogram_directory, row['id']))
        spectrogram = torch.tensor(spectrogram, dtype=torch.float32)

        label = torch.tensor([
            (row['freq_low'] - 60) / (300 - 60),
            (row['gain_low'] + 8) / 16,
            (np.log(row['q_low'])  - np.log(0.5)) / (np.log(2.0) - np.log(0.5)),
            (row['freq_mid'] - 300) / (3000 - 300),
            (row['gain_mid'] + 6) / 12,
            (np.log(row['q_mid'])  - np.log(0.5)) / (np.log(4.0) - np.log(0.5)),
            (row['freq_high'] - 3000) / (16000 - 3000),
            (row['gain_high'] + 8) / 16,
            (np.log(row['q_high']) - np.log(0.5)) / (np.log(2.0) - np.log(0.5)),
        ], dtype=torch.float32)        
        return spectrogram, label
    
