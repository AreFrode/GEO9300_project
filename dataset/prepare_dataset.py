"""
Script for extracting relevant variables from .nc file, as well as prepare a SIC column, and store as a .csv file easily read by Pandas
"""

# Author: Are Frode Kvanum
# Created 04-10-2025


from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from netCDF4 import Dataset


def read_and_merge(
    dataset: Dataset, time_string: str, trajectory_string: str, variable_string: str
) -> pd.DataFrame:
    reference_time = np.datetime64(dataset[time_string].units[14:])

    buoy_time = reference_time + dataset[time_string][:].astype("timedelta64[s]")

    buoy_df_list = []

    for traj, data, time in zip(
        dataset[trajectory_string],
        dataset[variable_string],
        buoy_time,
    ):
        buoy_df_list.append(
            pd.DataFrame(
                {f"{traj[8:]}_{variable_string}": data},
                index=time,
            )
            .resample("60min")
            .mean()
            .dropna()
        )

    full_df = pd.merge(
        buoy_df_list[0],
        buoy_df_list[1],
        how="outer",
        left_index=True,
        right_index=True,
    )

    for buoy_df in buoy_df_list[2:]:
        full_df = pd.merge(
            full_df, buoy_df, how="outer", left_index=True, right_index=True
        )

    return full_df


def main():
    path_data = "/home/arefk/phd/geo9300/GEO9300_project/dataset/"
    buoys_path = f"{path_data}2025_KVS_deployment_nonQCdata_v01.nc"

    buoys_data = Dataset(buoys_path, mode="r", format="NETCDF4")

    # Merge DataFrames into one, wuth multiindex both datetime and buoy_id

    full_temp_df = read_and_merge(buoys_data, "time_temp", "trajectory", "temp_air_raw")
    full_surf_df = read_and_merge(
        buoys_data, "time_temp", "trajectory", "temp_mlx_internal"
    )
    full_sii_df = read_and_merge(
        buoys_data, "time_temp", "trajectory", "temp_snow_ice_raw"
    )
    full_ice_df = read_and_merge(buoys_data, "time_temp", "trajectory", "temp_ice_raw")
    full_lat_df = read_and_merge(buoys_data, "time", "trajectory", "lat")
    full_lon_df = read_and_merge(buoys_data, "time", "trajectory", "lon")

    buoys_data.close()

    merged_df = pd.merge(full_lat_df, full_lon_df, left_index=True, right_index=True)

    temp_dfs = [full_temp_df, full_surf_df, full_sii_df, full_ice_df]

    for temp_df in temp_dfs:
        merged_df = pd.merge(merged_df, temp_df, left_index=True, right_index=True)

    # TODO: Add SIC column to merged_df here

    buoy_indices = []

    for col in merged_df.columns:
        if "lat" in col:
            buoy_ind = (
                col.split("_")[0] + "_" + col.split("_")[1] + "_" + col.split("_")[2]
            )
            buoy_indices.append(buoy_ind)

    reshaped_data = []
    for buoy_idx in buoy_indices:
        lat_col = f"{buoy_idx}_lat"
        lon_col = f"{buoy_idx}_lon"
        temp_col = f"{buoy_idx}_temp_air_raw"
        surf_col = f"{buoy_idx}_temp_mlx_internal"
        sii_col = f"{buoy_idx}_temp_snow_ice_raw"
        ice_col = f"{buoy_idx}_temp_ice_raw"

        reshaped_data.append(
            pd.DataFrame(
                {
                    "lat": merged_df[lat_col],
                    "lon": merged_df[lon_col],
                    "temp_air": merged_df[temp_col],
                    "temp_surf": merged_df[surf_col],
                    "temp_snow_ice": merged_df[sii_col],
                    "temp_ice": merged_df[ice_col],
                }
            ).assign(KVS_BUOY_IDX=buoy_idx)
        )

    full_latlontemp_df = pd.concat(reshaped_data)
    full_latlontemp_df.set_index(
        ["KVS_BUOY_IDX", full_latlontemp_df.index], inplace=True
    )

    full_latlontemp_df.to_csv(f"{path_data}prepared_buoy_data.csv")


if __name__ == "__main__":
    main()
