import os

import pandas as pd
import numpy as np

from xgboost import XGBRegressor, plot_importance

from sklearn.model_selection import train_test_split, GridSearchCV
from matplotlib import pyplot as plt
from ml_correction import add_cyclic_time


def main():
    #path_data = "../dataset/"
    path_data = "//kant.uio.no/geo-geofag-u1/matsip/PHD_BioGov/Courses/GEO9300_autumn2025/GEO9300_project/GEO9300_project/dataset/"
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

    X_train = np.concatenate(
        [X_train, X_train_time[['hour_sin', 'hour_cos', 'day_sin', 'day_cos']]], axis=1)

    X_val = np.concatenate(
        [X_val, X_val_time[['hour_sin', 'hour_cos', 'day_sin', 'day_cos']]], axis=1)

    param_grid = {
        'n_estimators': [100, 500],
        'max_depth': [3, 6, 9, 12],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0]
    }

    model = XGBRegressor(objective='reg:squarederror', random_state=1913)

    grid_search = GridSearchCV(
        estimator=model, param_grid=param_grid, cv=3, n_jobs=-1, verbose=1
    )
    grid_search.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    print("Best params:", grid_search.best_params_)
    model = grid_search.best_estimator_
    preds = model.predict(X_val)
    print(preds[:5])
    print(y_val[:5])

    os.makedirs('models/', exist_ok=True)

    model.save_model("models/xgboost_model.json")

    plot_importance(model, importance_type='gain')
    plt.savefig('xgboost_feature_importance.png')


if __name__ == "__main__":
    main()
