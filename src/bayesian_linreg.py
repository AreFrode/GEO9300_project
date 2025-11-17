import numpyro
import torch

import seaborn as sns
import numpy as np
import pandas as pd

from numpyro.diagnostics import hpdi
from numpyro.infer import MCMC, NUTS
from jax import random

from jax import numpy as jnp
from numpyro import distributions as dist
from matplotlib import pyplot as plt

from ml_correction import add_cyclic_time


def old_bayesian_model(x_data_, y_data_):
    s = pyro.sample("s", pyro.distributions.Normal(0, 1))

    b = pyro.sample("b", pyro.distributions.Normal(0., 1000.))

    theta = pyro.sample("theta", pyro.distributions.Gamma(1., 1.))

    mean = (b + x_data_ * s).squeeze(-1)
    pyro.deterministic("predictive_mean", mean)

    with pyro.plate("plate_x", len(x_data_)):
        pyro.sample("y", pyro.distributions.Normal(
            loc=mean, scale=torch.sqrt(1. / theta)), obs=y_data_)

    return mean


def bayesian_model(x_data_, y_data_):
    a = numpyro.sample("a", dist.Normal(0.0, 0.2))

    b = numpyro.sample("b", dist.Normal(0.0, 0.5))

    M = b * x_data_

    sigma = numpyro.sample("sigma", dist.Exponential(1.0))
    mu = a + M
    numpyro.sample("obs", dist.Normal(mu, sigma), obs=y_data_)


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


def standardize(x):
    return (x - x.mean()) / x.std()


def plot_regression(x, y, y_mean, y_hpdi):
    idx = jnp.argsort(x)
    arome_t2m = x[idx]
    mean = y_mean[idx]
    hpdi = y_hpdi[:, idx]
    residuals = y[idx]

    # Plot
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 6))
    ax.plot(arome_t2m, mean)
    ax.plot(arome_t2m, residuals, "o")
    ax.fill_between(arome_t2m, hpdi[0], hpdi[1], alpha=.3, interpolate=True)
    return fig, ax


def main():
    plt.style.use('bmh')

    path_data = "../dataset/"
    path_figs = "../figures/"
    buoys_df = pd.read_csv(
        f"{path_data}prepared_buoy_data.csv", index_col=[0, 1])

    # Keep buoy10 for testing
    kvs = buoys_df.loc[['KVS_SvalMIZ_03',
                        'KVS_SvalMIZ_07']].dropna().reset_index(level=0)

    kvs = kvs.set_index(pd.to_datetime(kvs.index))
    kvs['residuals'] = kvs['temp_air'] - kvs['arome_t2m']
    kvs['hour'] = kvs.index.hour
    kvs['doy'] = kvs.index.day_of_year

    X = kvs[['KVS_BUOY_IDX', 'arome_t2m', 'sic', 'hour', 'doy']]
    y = kvs[['residuals']]

    X_time = X[['hour', 'doy']]
    X = X.drop(['hour', 'doy'], axis=1)

    X_time = add_cyclic_time(X_time.copy())

    X = pd.concat(
        [X, X_time[['hour_sin', 'hour_cos', 'day_sin', 'day_cos']]], axis=1)

    idxs = ['KVS_BUOY_IDX', 'arome_t2m', 'sic',
            'hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'residuals']

    processed_kvs = pd.concat([X, y], axis=1)

    # sns.pairplot(processed_kvs, x_vars=idxs, y_vars=idxs,
    # hue="doy", palette='husl')
    # plt.savefig(f'{path_figs}pairplot_alldata.png')

    # sns.regplot(x='day_sin', y='residuals', data=processed_kvs)
    # plt.savefig(f"{path_figs}regplot_dsin_residuals.png")

    X["arome_t2m_scaled"] = X.arome_t2m.pipe(standardize)
    y.loc[:, 'residuals_scaled'] = y.residuals.pipe(standardize)

    # Start from source of randomness
    rng_key = random.PRNGKey(1913)
    rng_key, rng_key_ = random.split(rng_key)

    # Run NUTS
    kernel = NUTS(bayesian_model)

    num_samples = 2000
    mcmc = MCMC(kernel, num_warmup=1000, num_samples=num_samples)
    mcmc.run(
        rng_key_,
        X['arome_t2m_scaled'].values,
        y['residuals_scaled'].values
    )

    mcmc.print_summary()
    samples_1 = mcmc.get_samples()

    # Compute empirical posterior over mu
    posterior_mu = (
        jnp.expand_dims(samples_1["a"], -1)
        + jnp.expand_dims(samples_1["b"], -1) * X["arome_t2m_scaled"].values
    )

    mean_mu = jnp.mean(posterior_mu, axis=0)
    hpdi_mu = hpdi(posterior_mu, 0.9)
    fig, ax = plot_regression(
        X['arome_t2m_scaled'].values, y['residuals_scaled'].values, mean_mu, hpdi_mu)

    ax.set(
        xlabel="Forecasted T2M",
        ylabel="Resdiuals",
        title="Regression line with 90% CI"
    )

    fig.savefig(f"{path_figs}Posterior_dist_over_regression_parameters.png")
    ''' 
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
    '''


if __name__ == "__main__":
    main()
