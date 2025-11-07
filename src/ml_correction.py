import torch

import pandas as pd
import numpy as np

from tqdm import tqdm
from torch import nn as nn
from torch import functional as F

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# #### Prepare data

path_data = "../dataset/"
buoys_df = pd.read_csv(f"{path_data}prepared_buoy_data.csv", index_col=[0, 1])

kvs_10 = buoys_df.loc['KVS_SvalMIZ_10'].dropna()

X = kvs_10.drop(columns=['lat', 'lon', 'temp_air',
                'temp_surf', 'temp_snow_ice', 'temp_ice'])
y = kvs_10[['temp_air']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2)

X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=.2)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

y_scaler = StandardScaler()
y_train = y_scaler.fit_transform(y_train)
y_val = y_scaler.transform(y_val)
y_test = y_scaler.transform(y_test)


class BuoyDataset(torch.utils.data.Dataset):
    def __init__(self, features, targets):
        self.X = torch.tensor(features, dtype=torch.float32)
        self.y = torch.tensor(targets, dtype=torch.float32).view(-1, 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


train_dataset = BuoyDataset(X_train, y_train)
val_dataset = BuoyDataset(X_val, y_val)
test_dataset = BuoyDataset(X_test, y_test)

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=64, shuffle=True)
val_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64)

# #### Define model


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

# #### training loop


model = SimpleMLP(in_size=X.shape[1])
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.5,
    patience=10
)
criterion = nn.MSELoss()

best_val_loss = float('inf')
epochs = 100

prog_bar = tqdm(range(epochs))

for epoch in prog_bar:
    prog_bar.set_description(
        desc=f"Epoch: {epoch+1}, lr: {optimizer.param_groups[0]["lr"]}")
    model.train()
    total_loss = []

    # training
    for X, y in train_loader:
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
            val_pred = model(X)
            total_val_loss.append(criterion(val_pred, y).item())

    prog_bar.set_postfix({'loss': f"{np.mean(total_loss):.4f}",
                         "val_loss": f"{np.mean(total_val_loss):.4f}"})
    scheduler.step(np.mean(total_val_loss))


model.eval()
print("\n-----\n")

with torch.no_grad():

    test_rmse = []
    for X, y in test_loader:
        test_pred = y_scaler.inverse_transform(model(X))

        y = y_scaler.inverse_transform(y)

        for i, pred in enumerate(test_pred):
            se = (pred - y[i])**2
            test_rmse.append(np.sqrt(se))

print(f"{np.mean(test_rmse)}")
