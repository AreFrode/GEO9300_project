"""
Script for extracting relevant variables from .nc file, as well as prepare a SIC column, and store as a .csv file easily read by Pandas
"""

# Author: Are Frode Kvanum
# Created 04-10-2025


from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from netCDF4 import Dataset
from pyproj import CRS, Transformer


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

    # SIC column to merged_df here then merge with temp dfs
    new_columns = merged_df.columns.str.replace("lon", "sic")
    merged_df = merged_df.assign(
        **{
            new_col: merged_df[col]
            for new_col, col in zip(new_columns, merged_df.columns)
        }
    )

    # Define CARRA grid constants

    with Dataset(
        f"{path_data}CARRA/2025/04/CARRA_extracted_sic_20250427.nc", "r"
    ) as nc_carra_const:
        carra_x = nc_carra_const["x"][:]
        carra_y = nc_carra_const["y"][:]
        proj_carra = vars(nc_carra_const["Lambert_Conformal"])

    proj4_carra = f"+proj=lcc +lat_0={int(proj_carra['standard_parallel'])} +lon_0={int(proj_carra['longitude_of_central_meridian'])} +lat_1={int(proj_carra['latitude_of_projection_origin'])} +R={int(proj_carra['earth_radius'])} +y_0={proj_carra['false_northing']} +x_0={proj_carra['false_easting']}"

    crs_wgs84 = CRS.from_epsg(4326)
    crs_carra = CRS.from_proj4(proj4_carra)
    transformer_to_carra = Transformer.from_crs(crs_wgs84, crs_carra)

    for ts, row in merged_df.iterrows():
        # Extract current CARRA SIC field
        with Dataset(
            f"{path_data}CARRA/{ts.year}/{ts.month:02d}/CARRA_extracted_sic_{ts.year}{ts.month:02d}{ts.day:02d}.nc"
        ) as nc_carra:
            current_sic = nc_carra["ci"][0]

        for idx in row.index:
            if idx.endswith("_sic") and not pd.isna(row[idx]):
                row_lat = row[f"{idx[:-3]}lat"]
                row_lon = row[f"{idx[:-3]}lon"]

                row_x, row_y = transformer_to_carra.transform(row_lat, row_lon)

                x_idx = (np.abs(carra_x - row_x)).argmin()
                y_idx = (np.abs(carra_y - row_y)).argmin()

                merged_df.at[ts, idx] = current_sic[y_idx, x_idx]

    temp_dfs = [full_temp_df, full_surf_df, full_sii_df, full_ice_df]

    for temp_df in temp_dfs:
        merged_df = pd.merge(merged_df, temp_df, left_index=True, right_index=True)

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
        sic_col = f"{buoy_idx}_sic"

        reshaped_data.append(
            pd.DataFrame(
                {
                    "lat": merged_df[lat_col],
                    "lon": merged_df[lon_col],
                    "temp_air": merged_df[temp_col],
                    "temp_surf": merged_df[surf_col],
                    "temp_snow_ice": merged_df[sii_col],
                    "temp_ice": merged_df[ice_col],
                    "sic": merged_df[sic_col],
                }
            ).assign(KVS_BUOY_IDX=buoy_idx)
        )

    full_latlontemp_df = pd.concat(reshaped_data)
    full_latlontemp_df.set_index(
        ["KVS_BUOY_IDX", full_latlontemp_df.index], inplace=True
    )

    full_latlontemp_df.to_csv(f"{path_data}prepared_buoy_data.csv")

    test_df = pd.read_csv(f"{path_data}prepared_buoy_data.csv", index_col=[0, 1])
    print(test_df)


if __name__ == "__main__":
    main()
