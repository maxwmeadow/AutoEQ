import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from models.cnn import AutoEQ
from data.dataset import AutoEQDataset
import numpy as np
import argparse

DATA_DIRECTORY = 'data/processed'
SAVE_DIRECTORY = 'checkpoints/'
EPOCHS = 50
BATCH_SIZE = 32
LEARNING_RATE = 0.001
VALIDATION_SPLIT = 0.2
TEST_SPLIT = 0.2
SPLIT_SEED = 42

def eq_loss(prediction, target, device):
    weights = torch.tensor([1.0, 1.0, 1.5, 1.5, 1.5, 2.0, 2.0, 2.0, 3.0], device=device)

    low_activity = (target[:, 1] - 0.5).abs() * 2
    mid_activity = (target[:, 4] - 0.5).abs() * 2
    high_activity = (target[:, 7] - 0.5).abs() * 2

    mask = torch.ones_like(prediction)
    mask[:, 0] = low_activity
    mask[:, 2] = low_activity
    mask[:, 3] = mid_activity
    mask[:, 5] = mid_activity
    mask[:, 6] = high_activity
    mask[:, 8] = high_activity

    return (weights * mask * (prediction - target) ** 2).mean()

def train(data_directory, run_name):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device} for training")
    properties = torch.cuda.get_device_properties(device) if torch.cuda.is_available() else None
    gb_memory = properties.total_memory / (1024 ** 3) if properties else 'N/A'
    print(f"GPU Memory: {gb_memory} GB")

    dataset = AutoEQDataset(data_directory)
    n = len(dataset)
    n_test = int(n * TEST_SPLIT)
    n_val = int(n * VALIDATION_SPLIT)
    n_train = n - n_test - n_val
    generator = torch.Generator().manual_seed(SPLIT_SEED)
    train_set, val_set, test_set = random_split(dataset, [n_train, n_val, n_test], generator=generator)

    test_indices = list(test_set.indices)
    os.makedirs(SAVE_DIRECTORY, exist_ok=True)
    np.save(os.path.join(SAVE_DIRECTORY, f'test_indices_{run_name}.npy'), test_indices)

    training_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    validation_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    model = AutoEQ().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5, verbose=True)

    best_validation_loss = float('inf')

    for epoch in range(EPOCHS):
        model.train()
        training_loss = 0.0
        for spectrograms, labels in training_loader:
            spectrograms, labels = spectrograms.to(device), labels.to(device)

            if torch.rand(1).item() < 0.5:
                lambda_ = torch.distributions.Beta(0.4, 0.4).sample().to(device)
                index = torch.randperm(spectrograms.size(0), device=device)
                spectrograms = lambda_ * spectrograms + (1 - lambda_) * spectrograms[index]
                labels = lambda_ * labels + (1 - lambda_) * labels[index]

            optimizer.zero_grad()
            loss = eq_loss(model(spectrograms), labels, device)
            loss.backward()
            optimizer.step()
            training_loss += loss.item()

        model.eval()
        validation_loss = 0.0
        with torch.no_grad():
            for spectrograms, labels in validation_loader:
                spectrograms, labels = spectrograms.to(device), labels.to(device)
                validation_loss += eq_loss(model(spectrograms), labels, device).item()

        training_loss /= len(training_loader)
        validation_loss /= len(validation_loader)
        scheduler.step(validation_loss)

        print(f"Epoch {epoch+1}/{EPOCHS} - Training Loss: {training_loss:.4f} - Validation Loss: {validation_loss:.4f}")

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            torch.save(model.state_dict(), os.path.join(SAVE_DIRECTORY, f'best_model_{run_name}.pt'))
            print("New best model saved with validation loss: {:.4f}".format(best_validation_loss))

    print("Training complete. Best validation loss: {:.4f}".format(best_validation_loss))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default=DATA_DIRECTORY)
    parser.add_argument('--run', type=str, default='model')
    arguments = parser.parse_args()
    train(arguments.data, arguments.run)