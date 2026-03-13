import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

PARAMETER_NAMES = [
    'freq_low', 'gain_low', 'q_low',
    'freq_mid', 'gain_mid', 'q_mid',
    'freq_high', 'gain_high', 'q_high',
]

NORM_MIN   = np.array([60, -8, 0.5,  300, -6, 0.5,  3000, -8, 0.5], dtype=np.float32)
NORM_RANGE = np.array([240, 16, 1.5, 2700, 12, 3.5, 13000, 16, 1.5], dtype=np.float32)

all_acc, all_prec, all_rec, all_f1 = [], [], [], []

spectrogram = "data/processed/spectrograms"
labels = "data/processed/labels.csv"

df = pd.read_csv(labels)

X = []
for _, row in df.iterrows():
    spectrogram_ = np.load(f"data/processed/spectrograms/{row['id']}")
    X.append(spectrogram_.flatten())

X = np.array(X, dtype=np.float32)
y = df[PARAMETER_NAMES].values.astype(np.float32)
y = (y - NORM_MIN) / NORM_RANGE

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Fitting PCA")
pca = PCA(n_components=200)
X_train = pca.fit_transform(X_train)
X_test  = pca.transform(X_test)

# Discretize labels into 5 bins
binner = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='uniform')
y_train_binned = binner.fit_transform(y_train)
y_test_binned  = binner.transform(y_test)

model = MultiOutputClassifier(LogisticRegression(max_iter=100, solver='saga', tol=0.01))
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