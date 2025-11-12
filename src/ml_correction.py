import torch
import os

import pandas as pd
import numpy as np

from tqdm import tqdm
from torch import nn as nn
from torch import functional as F

from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn.preprocessing import StandardScaler
from joblib import dump


class BuoyDataset(torch.utils.data.Dataset):
    def __init__(self, features, targets):
        self.X = torch.tensor(features, dtype=torch.float32)
        self.y = torch.tensor(targets, dtype=torch.float32).view(-1, 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class SimpleMLP(nn.Module):
    def __init__(self, in_size=1, out_size=1, hidden_size=32):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, out_size),
        )

    def forward(self, x):
        return self.network(x)


class LessSimpleMLP(nn.Module):
    def __init__(self, in_size=1, out_size=1, hidden_sizes=[16, 32, 64]):
        super().__init__()
        layers = nn.ModuleList([])

        for h_size in hidden_sizes:
            layers.extend([
                nn.Linear(in_size, h_size),
                nn.BatchNorm1d(h_size),
                nn.ReLU(),
                nn.Dropout(0.1),
            ])
            in_size = h_size

        for h_size in hidden_sizes[-2::-1]:
            layers.extend([
                nn.Linear(in_size, h_size),
                nn.BatchNorm1d(h_size),
                nn.ReLU(),
                nn.Dropout(0.1),
            ])
            in_size = h_size

        layers.append(nn.Linear(in_size, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def add_cyclic_time(df):
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

    df['day_sin'] = np.sin(2 * np.pi * df['doy'] / 365)
    df['day_cos'] = np.cos(2 * np.pi * df['doy'] / 365)

    return df

def rmse(pred, target):
    return np.sqrt(np.mean((pred - target)**2))
# #### Prepare data


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    path_data = "../dataset/"
    buoys_df = pd.read_csv(
        f"{path_data}prepared_buoy_data.csv", index_col=[0, 1])

    # Keep buoy10 for testing
    kvs = buoys_df.loc[['KVS_SvalMIZ_03',
                        'KVS_SvalMIZ_07']].dropna().reset_index(level=0)

    kvs = kvs.set_index(pd.to_datetime(kvs.index))
    kvs['residuals'] = kvs['temp_air'] - kvs['arome_t2m']
    kvs['hour'] = kvs.index.hour
    kvs['doy'] = kvs.index.day_of_year

    X = kvs[['arome_t2m', 'sic', 'hour', 'doy']]
    y = kvs[['residuals']]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=.2, random_state=1913)

    X_train_time = X_train[['hour', 'doy']]
    X_train = X_train.drop(['hour', 'doy'], axis=1)

    X_val_time = X_val[['hour', 'doy']]
    X_val = X_val.drop(['hour', 'doy'], axis=1)

    X_train_time = add_cyclic_time(X_train_time.copy())
    X_val_time = add_cyclic_time(X_val_time.copy())

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    X_train = np.concatenate(
        [X_train, X_train_time[['hour_sin', 'hour_cos', 'day_sin', 'day_cos']]], axis=1)

    X_val = np.concatenate(
        [X_val, X_val_time[['hour_sin', 'hour_cos', 'day_sin', 'day_cos']]], axis=1)

    train_dataset = BuoyDataset(X_train, y_train.to_numpy())
    val_dataset = BuoyDataset(X_val, y_val.to_numpy())

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=64, shuffle=True, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=64, shuffle=True, pin_memory=True)

    rmse_val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=64, shuffle=False, pin_memory=True)

    # #### Define model
    criterion = nn.MSELoss()

    # #### training loop

    best_params = None
    best_rmse = float('inf')
    best_loss = float('inf')

    param_grid = {
        'lr': [0.001, 0.01, 0.1],
        'epochs': [20, 100, 200],
        'hidden_sizes': [[16, 32], [16, 32, 64], [16, 32, 64, 128], [32, 64, 128], [32, 64]]
    }

    validation_rmse = float('inf')

    for params in ParameterGrid(param_grid):

        model = LessSimpleMLP(
            in_size=X_train.shape[1], hidden_sizes=params['hidden_sizes']).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=10
        )

        prog_bar = tqdm(range(params['epochs']))
        total_val_loss = []

        for epoch in prog_bar:
            prog_bar.set_description(
                desc=f"Epoch: {epoch+1}, lr: {optimizer.param_groups[0]['lr']}")
            model.train()
            total_loss = []

            # training
            for X, y in train_loader:
                X = X.to(device)
                y = y.to(device)
                optimizer.zero_grad()
                preds = model(X)

                loss = criterion(preds, y)
                loss.backward()
                optimizer.step()
                total_loss.append(loss.item())

            # Validation
            model.eval()
            total_val_loss = []
            with torch.no_grad():
                for X, y in val_loader:
                    X = X.to(device)
                    y = y.to(device)

                    val_pred = model(X)
                    total_val_loss.append(criterion(val_pred, y).item())

            prog_bar.set_postfix({'loss': f"{np.mean(total_loss):.4f}",
                                  "val_loss": f"{np.mean(total_val_loss):.4f}",
                                  "val_rmse": f"{validation_rmse:.4f}"})
            scheduler.step(np.mean(total_val_loss))

        final_rmse_X = []
        model.eval()
        with torch.no_grad():
            for X, y in rmse_val_loader:
                X = X.to(device)

                val_pred = model(X).cpu()
                final_rmse_X.extend(val_pred)

        # print(final_rmse_X[:5])
        # print(kvs['arome_t2m'].shape)
        arome_t2m = scaler.inverse_transform(X_val[:, :2])[:,0]
        validation_rmse = rmse(arome_t2m + final_rmse_X, arome_t2m + y_val.to_numpy())
        prog_bar.set_postfix({'loss': f"{np.mean(total_loss):.4f}",
                                "val_loss": f"{np.mean(total_val_loss):.4f}",
                                "val_rmse": f"{validation_rmse:.4f}"})


        if validation_rmse < best_rmse:
            best_rmse = validation_rmse
            best_loss = np.mean(total_val_loss)
            best_params = params

    os.makedirs('models/', exist_ok=True)

    print(best_loss)
    print(best_rmse)
    print(best_params)

    exit()

    torch.save(model, 'models/diamond_dnn.pt')
    dump(scaler, 'models/X_train_scaler.bin', compress=True)


if __name__ == "__main__":
    main()
