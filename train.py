import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from models.cnn import AutoEQ
from data.dataset import AutoEQDataset

DATA_DIRECTORY = 'data/processed'
SAVE_DIRECTORY = 'checkpoints/'
EPOCHS = 50
BATCH_SIZE = 32
LEARNING_RATE = 0.001
VALIDATION_SPLIT = 0.2

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device} for training")
    properties = torch.cuda.get_device_properties(device) if torch.cuda.is_available() else None
    gb_memory = properties.total_memory / (1024 ** 3) if properties else 'N/A'
    print(f"GPU Memory: {gb_memory} GB")

    dataset = AutoEQDataset(DATA_DIRECTORY)
    validation_size = int(len(dataset) * VALIDATION_SPLIT)
    training_size = len(dataset) - validation_size
    training_set, validation_set = random_split(dataset, [training_size, validation_size])

    training_loader = DataLoader(training_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    validation_loader = DataLoader(validation_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    model = AutoEQ().to(device)
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5, verbose=True)

    os.makedirs(SAVE_DIRECTORY, exist_ok=True)
    best_validation_loss = float('inf')

    for epoch in range(EPOCHS):
        model.train()
        training_loss = 0.0
        for spectrograms, labels in training_loader:
            spectrograms, labels = spectrograms.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = loss_function(model(spectrograms), labels)
            loss.backward()
            optimizer.step()
            training_loss += loss.item()

        model.eval()
        validation_loss = 0.0
        with torch.no_grad():
            for spectrograms, labels in validation_loader:
                spectrograms, labels = spectrograms.to(device), labels.to(device)
                validation_loss += loss_function(model(spectrograms), labels).item()

        training_loss /= len(training_loader)
        validation_loss /= len(validation_loader)
        scheduler.step(validation_loss)

        print(f"Epoch {epoch+1}/{EPOCHS} - Training Loss: {training_loss:.4f} - Validation Loss: {validation_loss:.4f}")

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            torch.save(model.state_dict(), os.path.join(SAVE_DIRECTORY, 'best_model.pt'))
            print("New best model saved with validation loss: {:.4f}".format(best_validation_loss))

    print("Training complete. Best validation loss: {:.4f}".format(best_validation_loss))

if __name__ == "__main__":
    train()
