#!/bin/bash
# This is a smaller change

year=2025

echo "downloading AROME Arctic"

lustredir="/lustre/storeB/immutable/archive/projects/metproduction/DNMI_AROME_ARCTIC/${year}/"
localdir="/home/arefk/phd/geo9300/GEO9300_project/dataset/AROME_ARCTIC/${year}/"
rsync_options="-avz"

for MM in {04..07}; do
  targetdir="${localdir}${MM}/"
  mkdir -p "$targetdir"

  echo "Syncing month ${MM} from Lustre"
  if rsync $rsync_options lustre:"${lustredir}${MM}/**/arome_arctic_det_2_5km_*T00Z.nc" "$targetdir"; then
    echo "Successfully synced month ${MM} for year ${year}"
  fi

done
