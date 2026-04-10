# In this evaluation file I am using binning to conver the continuous regression
# outputs into discrete classes. This allows me to calculate the classification 
# metrics as required by the original project info.

import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from models.cnn import AutoEQ
from data.dataset import AutoEQDataset
from torch.utils.data import Subset

MODEL_PATH = 'checkpoints/best_model.pt'
DATA_DIRECTORY = 'data/processed'
BATCH_SIZE = 32
N_BINS = 5
NUM_CATEGORIES = 5

PARAMETER_NAMES = ['freq_low', 'gain_low', 'q_low', 'freq_mid', 'gain_mid', 'q_mid', 'freq_high', 'gain_high', 'q_high']

def evaluate():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device} for evaluation")
    properties = torch.cuda.get_device_properties(device) if torch.cuda.is_available() else None
    gb_memory = properties.total_memory / (1024 ** 3) if properties else 'N/A'
    print(f"GPU Memory: {gb_memory} GB")

    dataset = AutoEQDataset(DATA_DIRECTORY)
    test_indices = np.load(os.path.join('checkpoints', 'test_indices.npy'))
    test_set = Subset(dataset, test_indices)
    data_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    model = AutoEQ().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for spectrogram, labels in data_loader:
            spectrogram = spectrogram.to(device)
            predictions = model(spectrogram).cpu().numpy()
            all_predictions.append(predictions)
            all_labels.append(labels.numpy())

    all_labels = np.concatenate(all_labels, axis=0)
    all_predictions = np.concatenate(all_predictions, axis=0)

    mse = np.mean((all_predictions - all_labels) ** 2)
    print(f"Test MSE: {mse:.4f}")
    print_metrics(all_predictions, all_labels)

def print_metrics(all_predictions, all_labels):
    all_accuracy, all_precision, all_recall, all_f1 = [], [], [], []
    
    for i, name in enumerate(PARAMETER_NAMES):
        true_values = all_labels[:, i]
        prediction_values = all_predictions[:, i]

        category_boundaries = np.linspace(true_values.min(), true_values.max(), NUM_CATEGORIES + 1)

        true_categories = np.digitize(true_values, category_boundaries[1:-1])
        prediction_categories = np.digitize(prediction_values, category_boundaries[1:-1])
        
        prediction_categories = np.clip(prediction_categories, 0, NUM_CATEGORIES - 1)

        acc = accuracy_score(true_categories, prediction_categories)
        prec = precision_score(true_categories, prediction_categories, average='macro', zero_division=0)
        rec = recall_score(true_categories, prediction_categories, average='macro', zero_division=0)
        f1 = f1_score(true_categories, prediction_categories, average='macro', zero_division=0)

        all_accuracy.append(acc)
        all_precision.append(prec)
        all_recall.append(rec)
        all_f1.append(f1)

        print(f"{name} - Accuracy: {acc:.2f}, Precision: {prec:.2f}, Recall: {rec:.2f}, F1: {f1:.2f}")

    print("\nOverall Averages")
    print(f"Accuracy: {np.mean(all_accuracy):.2f}")
    print(f"Precision: {np.mean(all_precision):.2f}")
    print(f"Recall: {np.mean(all_recall):.2f}")
    print(f"F1 Score: {np.mean(all_f1):.2f}")

if __name__ == "__main__":
    evaluate()