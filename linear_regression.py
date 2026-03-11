import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor

spectrogram_folder = "data/processed/spectrograms"
labels_pathway = "data/processed/labels.csv"

PARAMETER_NAMES = [
    'freq_low', 'gain_low', 'q_low',
    'freq_mid', 'gain_mid', 'q_mid',
    'freq_high', 'gain_high', 'q_high',
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

X = np.array(X, dtype=np.float32)
y = df[PARAMETER_NAMES].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = MultiOutputRegressor(LinearRegression())
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

#sanity check for MSE
mse = np.mean((y_test - y_pred) ** 2)
print(f"Mean Squared Error (MSE): {mse:.4f}")

#saving the predictions for evaluate.py
np.save("linear_regression_predictions", y_pred)
np.save("linear_regression_labels", y_test)
