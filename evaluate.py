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
import argparse

MODEL_PATH = 'checkpoints/best_model.pt'
DATA_DIRECTORY = 'data/processed'
BATCH_SIZE = 32
N_BINS = 5
NUM_CATEGORIES = 5

PARAMETER_NAMES = ['freq_low', 'gain_low', 'q_low', 'freq_mid', 'gain_mid', 'q_mid', 'freq_high', 'gain_high', 'q_high']

def denormalize_values(predictions, labels):
    predictions_ = np.zeros_like(predictions)
    labels_ = np.zeros_like(labels)

    predictions_[:, 0] = predictions[:, 0] * (300 - 60) + 60
    labels_[:, 0] = labels[:, 0] * (300 - 60) + 60
    predictions_[:, 1] = predictions[:, 1] * 16 - 8
    labels_[:, 1] = labels[:, 1] * 16 - 8
    predictions_[:, 2] = np.exp(predictions[:, 2] * (np.log(2.0) - np.log(0.5)) + np.log(0.5))
    labels_[:, 2] = np.exp(labels[:, 2] * (np.log(2.0) - np.log(0.5)) + np.log(0.5))

    predictions_[:, 3] = predictions[:, 3] * (3000 - 300) + 300
    labels_[:, 3] = labels[:, 3] * (3000 - 300) + 300
    predictions_[:, 4] = predictions[:, 4] * 12 - 6
    labels_[:, 4] = labels[:, 4] * 12 - 6
    predictions_[:, 5] = np.exp(predictions[:, 5] * (np.log(4.0) - np.log(0.5)) + np.log(0.5))
    labels_[:, 5] = np.exp(labels[:, 5] * (np.log(4.0) - np.log(0.5)) + np.log(0.5))

    predictions_[:, 6] = predictions[:, 6] * (16000 - 3000) + 3000
    labels_[:, 6] = labels[:, 6] * (16000 - 3000) + 3000
    predictions_[:, 7] = predictions[:, 7] * 16 - 8
    labels_[:, 7] = labels[:, 7] * 16 - 8
    predictions_[:, 8] = np.exp(predictions[:, 8] * (np.log(2.0) - np.log(0.5)) + np.log(0.5))
    labels_[:, 8] = np.exp(labels[:, 8] * (np.log(2.0) - np.log(0.5)) + np.log(0.5))

    return predictions_, labels_

def print_tolerance(predictions, labels):
    predictions_, labels_ = denormalize_values(predictions, labels)
    print("\n Tolerance Metrics (Real World Metrics)")

    print("\n--Frequency (semitone tolerance)")
    for i, name in [(0, 'freq_low'), (3, 'freq_mid'), (6, 'freq_high')]:
        semitones = np.abs(12 * np.log2(predictions_[:, i] / labels_[:, i]))
        print(f"  {name:12s}  <=1st: {np.mean(semitones <= 1):.2f}  <=2st: {np.mean(semitones <= 2):.2f}  <=3st: {np.mean(semitones <= 3):.2f}")

    print("\n--Gain (dB tolerance)")
    for i, name in [(1, 'gain_low'), (4, 'gain_mid'), (7, 'gain_high')]:
        error = np.abs(predictions_[:, i] - labels_[:, i])
        print(f"  {name:12s}  <=1dB: {np.mean(error <= 1):.2f}  <=2dB: {np.mean(error <= 2):.2f}")

    print("\n--Q Factor (tolerance ratio)")
    for i, name in [(2, 'q_low'), (5, 'q_mid'), (8, 'q_high')]:
        ratio = np.maximum(predictions_[:, i] / labels_[:, i], labels_[:, i] / predictions_[:, i])
        print(f"  {name:12s}  <=1.5x: {np.mean(ratio <= 1.5):.2f}  <=2.0x: {np.mean(ratio <= 2.0):.2f}")

def evaluate(data_directory, model_path, run_name):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device} for evaluation")
    properties = torch.cuda.get_device_properties(device) if torch.cuda.is_available() else None
    gb_memory = properties.total_memory / (1024 ** 3) if properties else 'N/A'
    print(f"GPU Memory: {gb_memory} GB")

    dataset = AutoEQDataset(data_directory)
    test_indices = np.load(os.path.join('checkpoints', f'test_indices_{run_name}.npy'))
    test_set = Subset(dataset, test_indices)
    data_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    model = AutoEQ().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
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
    print_tolerance(all_predictions, all_labels)

def print_metrics(all_predictions, all_labels):
    all_accuracy, all_precision, all_recall, all_f1 = [], [], [], []
    
    for i, name in enumerate(PARAMETER_NAMES):
        true_values = all_labels[:, i]
        prediction_values = all_predictions[:, i]

        category_boundaries = np.linspace(0.0, 1.0, NUM_CATEGORIES + 1)

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
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default=DATA_DIRECTORY)
    parser.add_argument('--model', type=str, default=MODEL_PATH)
    parser.add_argument('--run', type=str, default='model')
    arguments = parser.parse_args()
    evaluate(arguments.data, arguments.model, arguments.run)