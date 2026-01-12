import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from FFanchorV2.FFanchors import FFanchors


class Rastrigin:
    @staticmethod
    def calculate(x, A=10):
        return A * len(x) + np.sum(x**2 - A * np.cos(2 * np.pi * x))
    
    @staticmethod
    def normalize(y, method='std'):
        if method == 'std':
            y_mean = np.mean(y, axis=0, keepdims=True)
            y_std = np.std(y, axis=0, keepdims=True)
            return (y - y_mean) / (y_std + 1e-9)
        raise ValueError(f"Unknown normalization method: {method}")
        
    @staticmethod
    def generate_data(n_samples=1024, n_features=1, A=10):
        x = np.linspace(-5.12, 5.12, n_samples).reshape(-1, n_features)
        y = np.array([Rastrigin.calculate(xi, A) for xi in x])
        y = np.reshape(y, (-1, 1))
        return Rastrigin.normalize(x, method='std'), Rastrigin.normalize(y, method='std')


class RastriginDataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.from_numpy(x).float()
        self.y = torch.from_numpy(y).float()
    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class StandardMLP(nn.Module):
    def __init__(self, in_features: int, hidden_dims: list[int], out_features: int = 1):
        super().__init__()
        layers = []
        prev_dim = in_features
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, out_features))
        self.net = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FFanchorsMLP(nn.Module):
    def __init__(self, ff_anchors_list: list[torch.Tensor], mlp_hidden_dims: list[int], mlp_out_features: int = 1, **ffanchors_kwargs):
        super().__init__()
        self.ffanchors = FFanchors(ff_anchors_list, **ffanchors_kwargs)
        vectors_dim = sum(len(anchors) for anchors in ff_anchors_list)
        self.mlp = StandardMLP(vectors_dim, mlp_hidden_dims, mlp_out_features)
    
    def forward(self, x: torch.Tensor, x_idx: torch.Tensor | None = None) -> torch.Tensor:
        vectors, _ = self.ffanchors(x, x_idx)
        return self.mlp(vectors)


def create_anchors_from_data(x_data: np.ndarray, n_anchors: int = 64) -> list[torch.Tensor]:
    anchors_list = []
    for i in range(x_data.shape[1]):
        feature_data = x_data[:, i]
        anchors = np.linspace(np.min(feature_data), np.max(feature_data), n_anchors)
        anchors_list.append(torch.tensor(anchors, dtype=torch.float32))
    return anchors_list


def plot_results(x, y_true, y_pred, title="Rastrigin Function Fitting", save_path=None):
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    sort_idx = np.argsort(x[:, 0])
    x_sorted = x[sort_idx, 0]
    y_true_sorted = y_true[sort_idx, 0]
    y_pred_sorted = y_pred[sort_idx, 0]
    ax.scatter(x_sorted, y_true_sorted, color='red', s=10, label='True', alpha=0.6)
    ax.scatter(x_sorted, y_pred_sorted, color='blue', s=5, label='Predicted', alpha=0.6)
    ax.set_xlabel('x (normalized)')
    ax.set_ylabel('y (normalized)')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    plt.close()


def test_rastrigin_fitting():
    torch.manual_seed(42)
    np.random.seed(42)
    
    A = 10.0
    dim = 1
    n_samples = 1024 * 16
    batch_size = 128
    n_anchors = 64
    mlp_hidden_dims = []
    learning_rate = 0.01
    n_epochs = 100
    
    print("=" * 70)
    print("Rastrigin Function Fitting - FFanchors V2 + Standard MLP")
    print("=" * 70)
    print(f"A={A}, dim={dim}, n_samples={n_samples}, batch_size={batch_size}")
    print(f"n_anchors={n_anchors}, mlp_hidden_dims={mlp_hidden_dims}, lr={learning_rate}, epochs={n_epochs}")
    
    x_np, y_np = Rastrigin.generate_data(n_samples=n_samples, n_features=dim, A=A)
    print(f"\nData: x.shape={x_np.shape}, y.shape={y_np.shape}")
    
    ff_anchors_list = create_anchors_from_data(x_np, n_anchors=n_anchors)
    vectors_dim = sum(len(a) for a in ff_anchors_list)
    print(f"Anchors: {[len(a) for a in ff_anchors_list]}, vectors_dim={vectors_dim}")
    
    model = FFanchorsMLP(ff_anchors_list, mlp_hidden_dims, 1, train_beta=False, train_anchors=False)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {total_params:,} parameters")
    
    dataset = RastriginDataset(x_np, y_np)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    
    print(f"\nTraining...")
    model.train()
    for epoch in range(n_epochs):
        epoch_losses = []
        for batch_x, batch_y in dataloader:
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())
        avg_loss = np.mean(epoch_losses)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch [{epoch+1:3d}/{n_epochs}], Loss: {avg_loss:.8f}")
    
    print(f"\nEvaluating...")
    model.eval()
    with torch.no_grad():
        x_torch = torch.from_numpy(x_np).float()
        y_pred = model(x_torch).numpy()
        mse = np.mean((y_np - y_pred) ** 2)
        rmse = np.sqrt(mse)
        relative_error = np.mean(np.square(y_np - y_pred) / (np.square(y_np) + 1e-3))
        print(f"  MSE: {mse:.8f}, RMSE: {rmse:.8f}, Relative Error: {relative_error:.8f}")
    
    output_dir = Path(__file__).parent.parent.parent / "experiments" / "rastrigin" / "FFanchorsV2-MLP"
    plot_results(x_np, y_np, y_pred, f"Rastrigin Fitting - FFanchors V2 + MLP\nEpochs: {n_epochs}, Loss: {avg_loss:.6f}", str(output_dir / "final_result.jpg"))
    
    print("=" * 70)
    print("Training completed!")
    print("=" * 70)


if __name__ == "__main__":
    test_rastrigin_fitting()
