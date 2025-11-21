# GEO9300_project
Project work for the course GEO9300 offered by the University of Oslo

## Repository Structure

```
GEO9300_project/
├── dataset/     # Raw and processed data files
├── biblio/      # Bibliography and reference material
├── src/         # Source code
├── src/models/  # Model weights and data scaling parameters
├── figures/     # Figures from development or report
├── report/      # Project report
└── README.md    # This file
```

## Data

*[Here we can describe the datasets used, sources, and processing]*

The `dataset/` directory contains:
- 2025_KVS_buoy17_deployment_nonQCdata_v01.nc
- 2025_KVS_deployment_nonQCdata_v01.nc
- SvalMIZ-25_SnowDepth-IceThickness.csv
- CARRA/
- AROME_ARCTIC/ (The AROME Arctic data is actually not available due to storage limitations, but can be provided upon request / accessed on [THREDDS](https://thredds.met.no/thredds/catalog/aromearcticarchive/2025/catalog.html))


## Requirements

In order to run the code, plase install the provided conda environment as follows
```bash
$ conda env create -f environment.yml
$ conda activate myenv
```

## Usage

*[How to run the code]*

## License

This project is licensed under the GPL-3.0 License - see the [LICENSE](LICENSE) for for details.

## Contact

- **Authors**: Mats Ippach, Are Frode Kvanum
- **Email**: m.r.ippach@geo.uio.no, arefk@met.no

*Last updated: September 2025*
