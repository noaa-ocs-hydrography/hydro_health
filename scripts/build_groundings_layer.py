import os
import geopandas as gpd
import pandas as pd
import requests
import pathlib


OUTPUTS = pathlib.Path(__file__).parents[1] / 'outputs'


class MarineCasualtiesEngine:
    """ETL Engine to extract data from an ArcGIS FeatureServer

    and write it locally as a GeoPackage (.gpkg).
    """

    def __init__(
        self,
        feature_service_url: str,
        filename: str = "reportable_marine_casualties.gpkg",
    ):
        self.feature_service_url = feature_service_url.rstrip("/")
        self.query_url = f"{self.feature_service_url}/query"
        self.output_dir = OUTPUTS
        self.filename = filename
        self.output_path = os.path.join(self.output_dir, self.filename)
        self.incident_types = [
            "Allision",
            "Collision",
            "Grounding",
            "Wave(s) Strikes/Impacts",
        ]

    def extract(self) -> gpd.GeoDataFrame:
        """Extracts all features from the ArcGIS FeatureServer using GeoJSON pagination."""
        print(f"[EXTRACT] Querying ArcGIS FeatureServer: {self.query_url}")

        offset = 0
        limit = 1000  # Standard batch size for FeatureServer queries
        gdfs = []

        while True:
            params = {
                "where": "1=1",
                "outFields": "*",
                "outSR": "4326",  # WGS84
                "f": "geojson",
                "resultOffset": offset,
                "resultRecordCount": limit,
            }

            response = requests.get(self.query_url, params=params)
            response.raise_for_status()
            data = response.json()

            features = data.get("features", [])
            if not features:
                break

            # Convert page to GeoDataFrame
            page_gdf = gpd.GeoDataFrame.from_features(features, crs="EPSG:4326")
            gdfs.append(page_gdf)

            print(f"  Downloaded batch of {len(features)} records (Offset: {offset})")

            # Check if we reached the end of records
            if len(features) < limit or data.get("exceededTransferLimit") is False:
                break

            offset += len(features)

        if not gdfs:
            raise ValueError("No features retrieved from FeatureServer.")

        # Combine all paginated batches into a single GeoDataFrame
        full_gdf = gpd.GeoDataFrame(
            pd.concat(gdfs, ignore_index=True), crs="EPSG:4326"
        )
        print(f"[EXTRACT] Total records extracted: {len(full_gdf)}")
        return full_gdf

    def load_local(
        self, gdf: gpd.GeoDataFrame, layer_name: str = "marine_casualties"
    ):
        """Saves the GeoDataFrame locally to a GeoPackage file."""
        print(f"[LOAD] Ensuring target directory exists: {self.output_dir}")
        os.makedirs(self.output_dir, exist_ok=True)

        print(f"[LOAD] Writing GeoPackage layer '{layer_name}' to: {self.output_path}")
        gdf.to_file(self.output_path, layer=layer_name, driver="GPKG")
        print(f"[LOAD] File successfully written to {self.output_path}")

    def transform(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
            """Filters the GeoDataFrame to include only specific Incident_Type values."""
            print(f"[TRANSFORM] Filtering records by Incident_Type...")

            if "Incident_Type" not in gdf.columns:
                raise KeyError(
                    "Attribute 'Incident_Type' was not found in the extracted dataset."
                )

            initial_count = len(gdf)

            # Perform boolean indexing filter
            filtered_gdf = gdf[
                gdf["Incident_Type"].isin(self.incident_types)
            ].copy()

            filtered_count = len(filtered_gdf)
            removed_count = initial_count - filtered_count

            print(f"  Target Incident Types: {self.incident_types}")
            print(
                f"  Filtered down from {initial_count} to {filtered_count} records (Removed {removed_count})."
            )

            return filtered_gdf

    def run(self, layer_name: str = "reportable_marine_casualties"):
        """Runs the full ETL execution lifecycle."""
        print("=== Starting ETL Process ===")
        raw_gdf = self.extract()
        transformed_gdf = self.transform(raw_gdf)
        self.load_local(transformed_gdf, layer_name=layer_name)
        print("=== ETL Process Finished Successfully ===")


# ==========================================
# Example Usage
# ==========================================
if __name__ == "__main__":
    FEATURE_LAYER_URL = (
        "https://services8.arcgis.com/6ldl6K67FkYzPtEE/ArcGIS/rest/services/"
        "Reportable_Marine_Casualties_Final_WFL1/FeatureServer/0"
    )

    # Instantiate engine with a local target path
    engine = MarineCasualtiesEngine(
        feature_service_url=FEATURE_LAYER_URL,
        filename="reportable_marine_casualties.gpkg",
    )

    engine.run(layer_name="marine_casualties_2026")