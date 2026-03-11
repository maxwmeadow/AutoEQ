import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor

spectrogram_folder = "data/processed/spectrograms"
labels_pathway = "data/processed/labels.csv"

PARAMETER_NAMES = [
    'freq_low', 'gain_low', 'Q_low',
    'freq_mid', 'gain_mid', 'Q_mid',
    'freq_high', 'gain_high', 'Q_high',
]

df = pd.read_csv(labels_pathway)

# rename with the actual column name of the CSV
filename= 'temp column name'

X = []
for _, row in df.iterrows():
    fn = str(row[filename])
    spectrogram_pathway = os.path.join(spectrogram_folder, fn)
    spectrogram = np.load(spectrogram_pathway)
    X.append(spectrogram.flatten())

X = np.array(X)
y = df[PARAMETER_NAMES].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = MultiOutputRegressor(LinearRegression())
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
