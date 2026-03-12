import numpy as np
import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.model_selection import train_test_split

PARAMETER_NAMES = [
    'freq_low', 'gain_low', 'Q_low',
    'freq_mid', 'gain_mid', 'Q_mid',
    'freq_high', 'gain_high', 'Q_high',
]

all_acc, all_prec, all_rec, all_f1 = [], [], [], []

spectrogram = "data/processed/spectrograms"
labels = "data/processed/labels.csv"

X = pd.read_csv(spectrogram).to_numpy()
y = pd.read_csv(labels).to_numpy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Discretize labels into 5 bins
binner = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='uniform')
y_train_binned = binner.fit_transform(y_train)
y_test_binned  = binner.transform(y_test)

model = MultiOutputClassifier(LogisticRegression(max_iter=1000))
model.fit(X_train, y_train_binned)
y_pred_binned = model.predict(X_test)

for i, name in enumerate(PARAMETER_NAMES):
    acc  = accuracy_score(y_test_binned[:, i], y_pred_binned[:, i])
    prec = precision_score(y_test_binned[:, i], y_pred_binned[:, i], average='macro', zero_division=0)
    rec  = recall_score(y_test_binned[:, i], y_pred_binned[:, i], average='macro', zero_division=0)
    f1   = f1_score(y_test_binned[:, i], y_pred_binned[:, i], average='macro', zero_division=0)

    all_acc.append(acc)
    all_prec.append(prec)
    all_rec.append(rec)
    all_f1.append(f1)

    print(f"  {name:<12} Acc={acc:.3f}  Prec={prec:.3f}  Rec={rec:.3f}  F1={f1:.3f}")

print(f"\n  AVERAGE      Acc={np.mean(all_acc):.3f}  Prec={np.mean(all_prec):.3f}  Rec={np.mean(all_rec):.3f}  F1={np.mean(all_f1):.3f}")