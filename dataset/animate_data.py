from datetime import datetime, timedelta

import cmocean
import matplotlib.animation as animation
import matplotlib.colors as colors
import numpy as np
import pandas as pd
from cartopy import crs as ccrs
from matplotlib import pyplot as plt
from matplotlib.cm import ScalarMappable
from mpl_toolkits.axes_grid1 import make_axes_locatable
from netCDF4 import Dataset


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

    buoy_kvs_10_idx = 9
    reference_time_temp = np.datetime64(dataset["time_temp"].units[14:])

    buoy_kvs_10_idx_time_temp = reference_time_temp + dataset["time_temp"][
        buoy_kvs_10_idx, :
    ].astype("timedelta64[s]")
    buoy_kvs_10_t1m = dataset["temp_air_raw"][buoy_kvs_10_idx, :]
    buoy_kvs_10_temp_df = (
        pd.DataFrame({"t1m": buoy_kvs_10_t1m}, index=buoy_kvs_10_idx_time_temp)
        .resample("60min")
        .mean()
    )

    reference_time_traj = np.datetime64(dataset["time"].units[14:])
    buoy_kvs_10_time_traj = reference_time_traj + dataset["time"][
        buoy_kvs_10_idx, :
    ].astype("timedelta64[s]")

    buoy_kvs_10_traj_df = (
        pd.DataFrame(
            {
                "lat": dataset["lat"][buoy_kvs_10_idx, :],
                "lon": dataset["lon"][buoy_kvs_10_idx, :],
            },
            index=buoy_kvs_10_time_traj,
        )
        .resample("60min")
        .mean()
    )

    full_df = pd.merge(
        buoy_kvs_10_traj_df, buoy_kvs_10_temp_df, left_index=True, right_index=True
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

        vmin_temp = full_df["t1m"].min()
        vmax_temp = full_df["t1m"].max()

        times = full_df.index.values

        # dummy_ice = ax.pcolormesh([[0,1]], [[0,1]], [[0,1]], cmap = cmocean.cm.ice, transform = data_proj, vmin = 0, vmax = 1)
        # ice_cbar = fig.colorbar(dummy_ice, ax = ax, label = 'SIC')

        # dummy_scatter = ax.scatter([], [], c=[], cmap = cmocean.cm.thermal, transform = data_proj, vmin = vmin_temp, vmax = vmax_temp)
        # divider = make_axes_locatable(ax)
        # cax = divider.append_axes("right", size="5%", pad=0.1, axes_class = plt.Axes)

        norm = colors.Normalize(vmin=vmin_temp, vmax=vmax_temp)
        sm = ScalarMappable(norm=norm, cmap=cmocean.cm.thermal)
        cbar = fig.colorbar(sm, ax=ax, label="Temperature [T1M]")
        # cbar.mappable = sm
        # ax.__last_mappable = None

        # dummy_scatter.remove()

        norm_ice = colors.Normalize(vmin=0, vmax=1)

        # dummy_ice.remove()
        # cb.remove()

        ice_pcolormesh = None
        scatter_plots = []

        # points_per_frame = max(1, len(full_df) // 50)
        points_per_frame = 6  # Hours

        def animate(frame):
            nonlocal ice_pcolormesh, scatter_plots

            if ice_pcolormesh is not None:
                ice_pcolormesh.remove()
                ice_pcolormesh = None

            start_idx = frame * points_per_frame
            end_idx = min(start_idx + points_per_frame, len(full_df))

            if start_idx < len(full_df):
                new_time = times[start_idx:end_idx][0].astype(datetime)
                print(f"{new_time=}")
                new_lons = full_df["lon"].iloc[start_idx:end_idx].values
                new_lats = full_df["lat"].iloc[start_idx:end_idx].values
                new_temps = full_df["t1m"].iloc[start_idx:end_idx].values

                # try:
                new_ice = Dataset(
                    f"{carra_path}{new_time.year}/{new_time.month:02d}/CARRA_extracted_sic_{new_time.strftime('%Y%m%d')}.nc"
                )["ci"][0, :]

                # except FileNotFoundError:
                # prev_time = new_time - timedelta(days = 1)
                # new_ice = Dataset(f"{amsr2_path}{prev_time.year}/{prev_time.month:02d}/RegridAMSR2_{prev_time.strftime('%Y%m%d')}.nc")['amsr2'][:]

                ice_pcolormesh = ax.pcolormesh(
                    lon,
                    lat,
                    new_ice,
                    cmap=cmocean.cm.ice,
                    transform=data_proj,
                    norm=norm_ice,
                    zorder=0,
                )

                new_scatter = ax.scatter(
                    new_lons,
                    new_lats,
                    c=new_temps,
                    cmap=cmocean.cm.thermal,
                    norm=norm,
                    transform=data_proj,
                    zorder=1,
                )

                scatter_plots.append(new_scatter)

            # total_points = min(end_idx, len(full_df))
            ax.set_title(
                f"Buoy KVS-10 Track - {new_time.strftime('%Y-%m-%d')}",
                fontsize=14,
                pad=20,
            )

            # return scatter_plots
            return (
                [ice_pcolormesh] + scatter_plots
                if ice_pcolormesh is not None
                else scatter_plots
            )

        total_frames = (len(full_df) + points_per_frame - 1) // points_per_frame

        anim = animation.FuncAnimation(
            fig, animate, frames=total_frames, interval=100, repeat=True, blit=False
        )

        return fig, anim

    fig, anim = create_animation()
    print("Writing .gif")
    anim.save("scatter_animation.gif", writer="pillow", fps=10)

    # print('Writing .mp4')
    # anim.save('scatter_animation.mp4', writer='ffmpeg', fps = 10)

    # Show figure
    # plt.tight_layout()
    # plt.show()


if __name__ == "__main__":
    main()
