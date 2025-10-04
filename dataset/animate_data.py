import cmocean
import matplotlib.animation as animation
import matplotlib.colors as colors
import numpy as np
import pandas as pd
from cartopy import crs as ccrs
from matplotlib import pyplot as plt
from matplotlib.cm import ScalarMappable
from netCDF4 import Dataset
from prepare_dataset import read_and_merge


def main():
    dataset = Dataset(
        "2025_KVS_deployment_nonQCdata_v01.nc", mode="r", format="NETCDF4"
    )

    # aa_path = "/lustre/storeB/immutable/archive/projects/metproduction/DNMI_AROME_ARCTIC"
    carra_path = "/home/arefk/phd/geo9300/GEO9300_project/dataset/CARRA/"

    # Barents is used because storeB is under maintenence and I know it works and have the correct dimensions wgen extracting lat and lon
    with Dataset(
        f"{carra_path}2025/05/CARRA_extracted_sic_20250501.nc", "r"
    ) as constants:
        lat = constants["lat"][:]
        lon = constants["lon"][:]

    # TODO:
    # Merge DataFrames into one, wuth mutliindex both datetime and buoy_id

    full_temp_df = read_and_merge(dataset, "time_temp", "trajectory", "temp_air_raw")
    full_lat_df = read_and_merge(dataset, "time", "trajectory", "lat")
    full_lon_df = read_and_merge(dataset, "time", "trajectory", "lon")
    full_latlon_df = pd.merge(
        full_lat_df, full_lon_df, left_index=True, right_index=True
    )

    full_latlontemp_df = pd.merge(
        full_latlon_df, full_temp_df, left_index=True, right_index=True
    )

    buoy_indices = []

    for col in full_latlontemp_df.columns:
        if "lat" in col or "lon" in col or "temp" in col:
            buoy_ind = (
                col.split("_")[0] + "_" + col.split("_")[1] + "_" + col.split("_")[2]
            )
            buoy_indices.append(buoy_ind)

    reshaped_data = []
    for buoy_idx in buoy_indices:
        lat_col = f"{buoy_idx}_lat"
        lon_col = f"{buoy_idx}_lon"
        temp_col = f"{buoy_idx}_temp_air_raw"

        reshaped_data.append(
            pd.DataFrame(
                {
                    "lat": full_latlontemp_df[lat_col],
                    "lon": full_latlontemp_df[lon_col],
                    "t1m": full_latlontemp_df[temp_col],
                }
            ).assign(KVS_BUOY_IDX=buoy_idx)
        )

    full_latlontemp_df = pd.concat(reshaped_data)
    full_latlontemp_df.set_index(
        ["KVS_BUOY_IDX", full_latlontemp_df.index], inplace=True
    )

    def create_animation():
        map_proj = ccrs.NorthPolarStereo(central_longitude=10)
        data_proj = ccrs.PlateCarree()
        fig, ax = plt.subplots(figsize=(10, 6), subplot_kw={"projection": map_proj})

        ax.set_extent([-1, 25, 79, 82], crs=data_proj)
        ax.coastlines()
        gl = ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)

        gl.bottom_labels = False
        gl.right_labels = False

        vmin_temp = full_latlontemp_df["t1m"].quantile(0.05)
        vmax_temp = full_latlontemp_df["t1m"].quantile(0.95)

        times = sorted(full_latlontemp_df.index.get_level_values(1).unique())

        norm = colors.Normalize(vmin=vmin_temp, vmax=vmax_temp)
        sm = ScalarMappable(norm=norm, cmap=cmocean.cm.thermal)
        cbar = fig.colorbar(sm, ax=ax, label="Temperature [T1M]")

        norm_ice = colors.Normalize(vmin=0, vmax=1)

        ice_pcolormesh = None
        all_scatter_plots = []

        # Color map for different buoys
        unique_kvs_indices = full_latlontemp_df.index.get_level_values(0).unique()
        buoy_colors = plt.cm.tab10(np.linspace(0, 1, len(unique_kvs_indices)))
        buoy_color_map = dict(zip(unique_kvs_indices, buoy_colors))

        # Set timesteps per frame
        timesteps_per_frame = 6  # Hours

        def animate(frame):
            nonlocal ice_pcolormesh, all_scatter_plots

            if ice_pcolormesh is not None:
                ice_pcolormesh.remove()
                ice_pcolormesh = None

            start_idx = frame * timesteps_per_frame
            end_idx = min(
                start_idx + timesteps_per_frame,
                len(full_latlontemp_df.index.get_level_values(1).unique()),
            )

            new_scatter_plots = []  # New scatter plots for this frame

            if start_idx < len(times):
                time_range = times[start_idx:end_idx]

                print(f"Frame {frame}, Time range: {time_range[0]} to {time_range[-1]}")

                frame_data = full_latlontemp_df[
                    full_latlontemp_df.index.get_level_values(1).isin(time_range)
                ]

                if not frame_data.empty:
                    clean_data = frame_data.dropna()

                    if not clean_data.empty:
                        # Get buoy indices for edge colors
                        buoy_indices = clean_data.index.get_level_values(0)
                        edge_colors = [buoy_color_map[idx] for idx in buoy_indices]

                        # plot all data points in frame at once
                        scatter = ax.scatter(
                            clean_data["lon"],
                            clean_data["lat"],
                            c=clean_data["t1m"],
                            cmap=cmocean.cm.thermal,
                            norm=norm,
                            transform=data_proj,
                            edgecolors=edge_colors,
                            linewidths=0.5,
                            zorder=3,
                        )

                        new_scatter_plots.append(scatter)
                        all_scatter_plots.append(scatter)

                        max_scatter_plots = 10
                        while len(all_scatter_plots) > max_scatter_plots:
                            old_scatter = all_scatter_plots.pop(0)
                            old_scatter.remove()

                        for i, sc in enumerate(all_scatter_plots):
                            alpha = (i + 1) / len(all_scatter_plots)
                            sc.set_alpha(alpha)

                new_ice = Dataset(
                    f"{carra_path}{time_range[-1].year}/{time_range[-1].month:02d}/CARRA_extracted_sic_{time_range[-1].strftime('%Y%m%d')}.nc"
                )["ci"][0, :]

                ice_pcolormesh = ax.pcolormesh(
                    lon,
                    lat,
                    new_ice,
                    cmap=cmocean.cm.ice,
                    transform=data_proj,
                    norm=norm_ice,
                    zorder=0,
                )

                ax.set_title(
                    f"Buoy KVS-10 Track - {time_range[-1].strftime('%Y-%m-%d')}",
                    fontsize=14,
                    pad=20,
                )

            # return scatter_plots
            artists = new_scatter_plots

            if ice_pcolormesh is not None:
                artists.append(ice_pcolormesh)

            return artists

        total_frames = (
            len(full_latlontemp_df.index.get_level_values(1).unique())
            + timesteps_per_frame
            - 1
        ) // timesteps_per_frame

        anim = animation.FuncAnimation(
            fig, animate, frames=total_frames, interval=100, repeat=True, blit=False
        )

        return fig, anim

    fig, anim = create_animation()
    print("Writing .gif")
    anim.save("scatter_animation.gif", writer="pillow", fps=10)

    # print("Writing .mp4")
    # anim.save("scatter_animation.mp4", writer="ffmpeg", fps=10)

    # Show figure
    # plt.tight_layout()
    # plt.show()


if __name__ == "__main__":
    main()
