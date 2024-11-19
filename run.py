from models.GViT import GViT
from dataset.EEGEyeNet import EEGEyeNetDataset
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import numpy as np
from helper_functions import split

# Hyperparameter
batch_size = 64
n_epoch = 15
learning_rate = 1e-3
threshold = 0.7  # threshold for adjacency matrix

# loss function
criterion = nn.MSELoss()

optimizer = torch.optim.AdamW
scheduler_step_size = 6
scheduler_gamma = 0.1

dataset_path = './dataset/Position_task_with_dots_synchronised_min.npz'


def compute_adjacency_matrix(dataset, threshold):
    num_channels = 129
    correlation_sum = np.zeros((num_channels, num_channels))

    for i in range(len(dataset)):
        eeg_sample, _, _ = dataset[i]
        eeg_signal = eeg_sample.squeeze(0).numpy()  # (129, 500)
        correlation_matrix = np.corrcoef(eeg_signal)
        correlation_sum += correlation_matrix

    correlation_avg = correlation_sum / len(dataset)
    adj_matrix = np.where(np.abs(correlation_avg) > threshold, correlation_avg, 0)
    np.fill_diagonal(adj_matrix, 0)
    return adj_matrix


def train(model, dataset, optimizer, criterion, scheduler=None, batch_size=64, patience=5):
    best_val_loss = float('inf')
    patience_counter = 0

    train_indices, val_indices, test_indices = split(dataset.trainY[:, 0], 0.7, 0.15, 0.15)
    train = Subset(dataset, indices=train_indices)
    val = Subset(dataset, indices=val_indices)
    test = Subset(dataset, indices=test_indices)

    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model = model.to(device)
    criterion = criterion.to(device)

    for epoch in range(n_epoch):
        model.train()
        train_loss = 0.0

        for inputs, targets, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{n_epoch}"):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), targets.squeeze())
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        print(f"Epoch {epoch+1}, Train Loss: {train_loss / len(train_loader):.4f}")

        model.eval()
        with torch.no_grad():
            val_loss = sum(criterion(model(inputs.to(device)).squeeze(), targets.to(device).squeeze()).item()
                           for inputs, targets, _ in val_loader) / len(val_loader)
            print(f"Epoch {epoch+1}, Val Loss: {val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(model.state_dict(), "best_model.pt")
                print("Saved best model.")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print("Early stopping triggered.")
                    break

            test_loss = sum(criterion(model(inputs.to(device)).squeeze(), targets.to(device).squeeze()).item()
                            for inputs, targets, _ in test_loader) / len(test_loader)
            print(f"Epoch {epoch+1}, Test Loss: {test_loss:.4f}")

        if scheduler is not None:
            scheduler.step()



if __name__ == "__main__":
    print("Loading dataset...")
    dataset = EEGEyeNetDataset(dataset_path)
    adj_matrix = compute_adjacency_matrix(dataset, threshold)
    print("Adjacency matrix computed.")

    print("Initializing model...")
    model = GViT(adj_matrix)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step_size, gamma=scheduler_gamma)

    print("Starting training...")
    train(model, dataset, optimizer, criterion, scheduler, batch_size=batch_size)
