# Machine-Learning Seabed Elevation-Change Model

This package translates the finalized R seabed elevation-change workflow into a cloud-oriented Python workflow using XGBoost, Dask, GeoParquet, Rasterio/GDAL, Cloud Optimized GeoTIFFs, VRT mosaics, and S3FS.

## Package files

- `machine_learning_seabed_elevation_change_model.py` — complete executable workflow and training guide.
- `machine_learning_seabed_elevation_change_model.example.yaml` — fully commented configuration.
- `requirements-machine-learning-seabed-model.txt` — Python dependencies.

## Scientific formulation retained

The production model predicts future bathymetry directly:

```text
bathy_t + environmental predictors -> predicted_bathy_t1
```

Change products are then derived:

```text
predicted_delta_bathy = predicted_bathy_t1 - bathy_t
predicted_delta_rate  = predicted_delta_bathy / interval_years
```

Predictor missing values remain as `NaN` and are passed to XGBoost. Rows are removed from fitting only when essential response, coordinate, starting-bathymetry, delta, or interval fields are unusable. The separate `full_tile_data` and `subgrid_data` structure is retained.

## July 31 seam-safe revision

The authoritative master-grid polygon now controls every tiled LOCAL and GLOBAL raster footprint.

For each output tile, the workflow:

1. transforms the official grid polygon into the template-raster CRS;
2. crops a window aligned to the common template origin and resolution;
3. rasterizes finite point values within that window;
4. masks every cell outside the exact grid polygon;
5. writes explicit NoData (`-9999` by default);
6. creates a tiled/compressed COG and internal overviews.

Prediction GeoParquet rows may extend slightly outside the nominal tile polygon, but those rows can no longer create valid raster cells outside the official tile. Adjacent COGs therefore do not contain competing valid overlap cells, preventing the wrong tile from drawing over the correct tile in a VRT.

## Integrated full-extent LOCAL workflow

Full-extent local prediction is now part of the standard `local` stage.

The order is:

1. train every available tile-native LOCAL model;
2. produce immediate tile-native predictions where prediction data are present;
3. identify all trained local ensembles for each year pair;
4. assign each prediction tile to the nearest trained model using an interior representative point;
5. retain existing tile-native outputs by default, or overwrite them when configured;
6. fill prediction-only tiles;
7. write failure details by year pair;
8. optionally build LOCAL VRT mosaics.

Relevant YAML settings:

```yaml
deploy_local_full_extent: true
  # Standard behavior. Set false only for a deliberate tile-native-only test.

overwrite_existing_local: false
  # false preserves immediate tile-native predictions.
  # true rebuilds all full-extent predictions from assigned local ensembles.

write_full_extent_spatial_outputs: true
  # Writes GeoParquet and seam-safe COG outputs.

build_vrts_after_local: true
  # Builds LOCAL VRTs after the complete local extent exists.
```

The standalone `full-extent-local` stage remains available for rerunning deployment without retraining models.

## VRT creation

The `vrt` stage creates:

- LOCAL training mosaics under `training_root/VRT/`;
- LOCAL prediction mosaics under `prediction_root/VRT/`;
- GLOBAL prediction mosaics under `prediction_root/VRT/GLOBAL/`;
- optional multiband all-year VRT stacks when more than one year pair exists.

VRT creation explicitly sets source and destination NoData to the configured numeric NoData value. It requires GDAL Python bindings or the `gdalbuildvrt` command-line program.

## Typical commands

Run local training, immediate predictions, automatic full-extent deployment, and LOCAL VRT creation:

```bash
python machine_learning_seabed_elevation_change_model.py \
  --config machine_learning_seabed_elevation_change_model.example.yaml \
  --stage local
```

Rerun only full-extent local deployment:

```bash
python machine_learning_seabed_elevation_change_model.py \
  --config machine_learning_seabed_elevation_change_model.example.yaml \
  --stage full-extent-local
```

Train and apply global models:

```bash
python machine_learning_seabed_elevation_change_model.py \
  --config machine_learning_seabed_elevation_change_model.example.yaml \
  --stage global-train

python machine_learning_seabed_elevation_change_model.py \
  --config machine_learning_seabed_elevation_change_model.example.yaml \
  --stage global-predict
```

Build or refresh all VRTs:

```bash
python machine_learning_seabed_elevation_change_model.py \
  --config machine_learning_seabed_elevation_change_model.example.yaml \
  --stage vrt
```

Run the complete workflow:

```bash
python machine_learning_seabed_elevation_change_model.py \
  --config machine_learning_seabed_elevation_change_model.example.yaml \
  --stage all
```

The `all` sequence is now:

```text
local (including full extent) -> global-train -> global-predict -> vrt
```

It no longer runs a duplicate standalone full-extent stage after `local`.

## Recommended parity test

Start with one familiar tile and one year pair:

```yaml
year_pairs: ["2004_2006"]
tile_ids: ["BH4S556X_3"]
n_boot: 2
```

Compare R and Python for:

- locked predictor names and order;
- full, essential, and fitting row counts;
- predictor NA summaries;
- sample-weight summaries;
- spatial CV folds and selected boosting rounds;
- training metrics and delta distributions;
- erosion/deposition proportions;
- COG CRS, resolution, transform, NoData, and pixel dimensions;
- absence of valid raster values outside each official grid polygon;
- VRT seams at neighboring tile boundaries.

## Environment note

The script passes Python syntax compilation in the supplied package. A full import/run requires the packages in the requirements file, including Dask and the geospatial GDAL/Rasterio stack.
