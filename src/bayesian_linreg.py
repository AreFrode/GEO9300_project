import numpy as np
import pandas as pd
import pyro
import torch

from ml_correction import add_cyclic_time


def bayesian_model(x_data_, y_data_):
    s = pyro.sample("s", pyro.distributions.Normal(0, 1))

    b = pyro.sample("b", pyro.distributions.Normal(0., 1000.))

    theta = pyro.sample("theta", pyro.distributions.Gamma(1., 1.))

    mean = (b + x_data_ * s).squeeze(-1)
    pyro.deterministic("predictive_mean", mean)

    with pyro.plate("plate_x", len(x_data_)):
        pyro.sample("y", pyro.distributions.Normal(
            loc=mean, scale=torch.sqrt(1. / theta)), obs=y_data_)

    return mean


def train(x_data, y_data, model, num_iterations=1500):
    optim = pyro.optim.Adam({"lr": 0.1})
    param = {}

    guide = pyro.infer.autoguide.guides.AutoNormal(model)

    svi = pyro.infer.SVI(model, guide, optim,
                         loss=pyro.infer.Trace_ELBO())
    pyro.clear_param_store()

    for j in range(num_iterations):
        loss = svi.step(x_data, y_data)
        if j % 500 == 0:
            print("[iteration %04d] loss: %.4f" %
                  (j, loss / len(x_data)))

        param = {'vi_parameters': pyro.get_param_store().get_state(),
                 'guide': guide}

    return param


def main():
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

    X_time = X[['hour', 'doy']]
    X = X.drop(['hour', 'doy'], axis=1)

    X_time = add_cyclic_time(X_time.copy())

    X = np.concatenate(
        [X, X_time[['hour_sin', 'hour_cos', 'day_sin', 'day_cos']]], axis=1)

    idxs = ['arome_t2m', 'sic', 'hour_sin', 'hour_cos', 'day_sin', 'day_cos']

    x_data = {}
    for idx, val in enumerate(idxs):
        x_data[idxs[idx]] = torch.tensor(
            X[:, idx].reshape(-1, 1), dtype=torch.float32)

    y_data = {'residuals': torch.tensor(
        y['residuals'].values, dtype=torch.float32)}

    print(x_data)
    print(y_data)

    param = train(x_data['arome_t2m'], y_data['residuals'], bayesian_model)

    # Print the parameters
    header = f"\nThe parameters for bayesion linreg:"
    print(header)
    print(f"-"*len(header))
    for name, value in param['vi_parameters']['params'].items():
        print(f"{name:<25}: {value.detach().numpy()}")


if __name__ == "__main__":
    main()
