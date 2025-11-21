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

The `dataset/` directory contains:
- 2025_KVS_buoy17_deployment_nonQCdata_v01.nc
- 2025_KVS_deployment_nonQCdata_v01.nc
- SvalMIZ-25_SnowDepth-IceThickness.csv
- CARRA/
- AROME_ARCTIC/            # The AROME Arctic data is not available due to storage limitations, but can be provided upon request / accessed on [THREDDS](https://thredds.met.no/thredds/catalog/aromearcticarchive/2025/catalog.html)
- prepared_buoy_data.csv   # Dataset with Buoys, AROME Arctic and CARRA unified
- buoy_10_w_models.csv     # Test dataset with results from developed models

## Requirements

In order to run the code, please install the provided conda environment as follows
```bash
$ conda env create -f environment.yml
$ conda activate myenv
```

## Usage

The dataset is generated using the `dataset/prepare_dataset.py` script.

`src/simple_ml_correction.py` and `src/ml_correction.py` trains neural network models.
`src/train_xgboost*.py` trains xgboost models as denoted by name of script.

`src/ML_correction.ipynb` constructs and saves the test dataset with bias correction models, also computes RMSE and Feature importance for XGBoost.
`src/evaluation_finalfinal.ipynb` generates report plots, conducts KS experiment and performs OLS fit



## License

This repository is licensed under the GPL-3.0 License - see the [LICENSE](LICENSE) for for details.

## Contact

- **Authors**: Mats Ippach, Are Frode Kvanum
- **Email**: m.r.ippach@geo.uio.no, arefk@met.no

*Last updated: November 2025*
