import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluate import print_metrics

spectrogram_folder = "data/processed/spectrograms"
labels_pathway = "data/processed/labels.csv"

PARAMETER_NAMES = [
    'freq_low', 'gain_low', 'q_low',
    'freq_mid', 'gain_mid', 'q_mid',
    'freq_high', 'gain_high', 'q_high',
]

NORM_MIN = np.array([60, -8, 0.5,  300, -6, 0.5,  3000, -8, 0.5], dtype=np.float32)
NORM_RANGE = np.array([240, 16, 1.5, 2700, 12, 3.5, 13000, 16, 1.5], dtype=np.float32)

df = pd.read_csv(labels_pathway)

# rename with the actual column name of the CSV
filename= 'id'

X = []
for _, row in df.iterrows():
    fn = str(row[filename])
    spectrogram_pathway = os.path.join(spectrogram_folder, fn)
    spectrogram = np.load(spectrogram_pathway)
    X.append(spectrogram.flatten())

X = np.array(X, dtype=np.float32)
y = df[PARAMETER_NAMES].values
y = (y.astype(np.float32) - NORM_MIN) / NORM_RANGE


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = MultiOutputRegressor(LinearRegression())
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

#sanity check for MSE
mse = np.mean((y_test - y_pred) ** 2)
print(f"Mean Squared Error (MSE): {mse:.4f}")

print_metrics(y_pred, y_test)


