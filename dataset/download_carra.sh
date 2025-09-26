#!/bin/bash

year=2025

echo "downloading Carra"

atosdir="/ec/res4/scratch/nor1878/carra_retrievals/${year}/"
localdir="/home/arefk/phd/geo9300/GEO9300_project/dataset/CARRA/${year}/"
rsync_options="-avz"

for MM in {04..07}; do
  targetdir="${localdir}${MM}/"
  mkdir -p "$targetdir"

  echo "Syncing month ${MM} from ECMWF HPC"
  if rsync $rsync_options hpc-login:"${atosdir}${MM}/" "$targetdir"; then
    echo "Successfully synced month ${MM} for year ${year}"
  fi

done
