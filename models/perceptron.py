import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import KBinsDiscretizer

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluate import print_metrics

SPECTROGRAM_FOLDER = "data/processed/spectrograms"
LABELS_PATH = "data/processed/labels.csv"
N_BINS = 5

PARAMETER_NAMES = ['freq_low', 'gain_low', 'q_low', 'freq_mid', 'gain_mid', 'q_mid', 'freq_high', 'gain_high', 'q_high']
NORM_MIN = np.array([60, -8, 0.5,  300, -6, 0.5,  3000, -8, 0.5], dtype=np.float32)
NORM_RANGE = np.array([240, 16, 1.5, 2700, 12, 3.5, 13000, 16, 1.5], dtype=np.float32)

df = pd.read_csv(LABELS_PATH)

X = []
for _, row in df.iterrows():
    spectrogram = np.load(os.path.join(SPECTROGRAM_FOLDER, str(row['id'])))
    X.append(spectrogram.flatten())

X = np.array(X, dtype=np.float32)
y_raw = df[PARAMETER_NAMES].values.astype(np.float32)
y_norm = (y_raw - NORM_MIN) / NORM_RANGE

discretizer = KBinsDiscretizer(n_bins=N_BINS, encode='ordinal', strategy='uniform')
y_binned = discretizer.fit_transform(y_norm).astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y_binned, test_size=0.2, random_state=404)

print("Training perceptron...")
model = MultiOutputClassifier(Perceptron(max_iter=1000, tol=1e-3, random_state=0))
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print_metrics(y_pred, y_test)
