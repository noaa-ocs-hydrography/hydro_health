import boto3
import geopandas as gpd
import pandas as pd
import folium
from folium import Element, LayerControl
import os
import tempfile
import pathlib

# Assuming these exist in your environment
from hydro_health.helpers.tools import get_config_item

INPUTS = pathlib.Path(__file__).parents[3] / 'inputs'
OUTPUTS = pathlib.Path(__file__).parents[3] / 'outputs'

class RasterMaskChecker:
    def __init__(self):
        self.s3 = boto3.client('s3')
        self.bucket = get_config_item('SHARED', 'OUTPUT_BUCKET')
        self.prefix = 'testing/ER_3/model_variables/Prediction/raw/DigitalCoast'

    def _download_shapefile_set(self, shp_key):
        temp_dir = tempfile.mkdtemp()
        base_path = shp_key.rsplit('.', 1)[0]
        extensions = ['.shp', '.shx', '.dbf', '.prj']
        local_shp_path = ""
        
        for ext in extensions:
            key = f"{base_path}{ext}"
            local_file = os.path.join(temp_dir, os.path.basename(key))
            try:
                self.s3.download_file(self.bucket, key, local_file)
                if ext == '.shp': local_shp_path = local_file
            except Exception:
                continue 
        return local_shp_path

    def load_s3_shapefiles(self, simplify_tolerance=0.0001):
        """Loads the dissolved shapefiles from S3."""
        paginator = self.s3.get_paginator('list_objects_v2')
        all_gdfs = []

        for page in paginator.paginate(Bucket=self.bucket, Prefix=self.prefix):
            for obj in page.get('Contents', []):
                key = obj['Key']
                if key.endswith('_dis.shp') and 'NCEI' not in key:
                    path = self._download_shapefile_set(key)
                    if path:
                        gdf = gpd.read_file(path).to_crs(epsg=4326)
                        gdf['geometry'] = gdf.simplify(tolerance=simplify_tolerance, preserve_topology=True)
                        gdf['filename'] = os.path.basename(key)
                        all_gdfs.append(gdf[['filename', 'geometry']])
        
        return pd.concat(all_gdfs, ignore_index=True) if all_gdfs else None

    def load_ecoregions(self, gpkg_path, layer_name='EcoRegions_50m'):
        """Loads the specific EcoRegions layer from the local GeoPackage."""
        if not os.path.exists(gpkg_path):
            print(f"Warning: GeoPackage not found at {gpkg_path}")
            return None
        eco_gdf = gpd.read_file(gpkg_path, layer=layer_name).to_crs(epsg=4326)
        return eco_gdf

    def create_interactive_map(self, output_file=f"{pathlib.Path(OUTPUTS) / 'overlap_check.html'}"):
        # 1. Load Data
        s3_gdf = self.load_s3_shapefiles()
        eco_gdf = self.load_ecoregions(str(INPUTS / get_config_item('SHARED', 'MASTER_GRIDS')))

        if s3_gdf is None and eco_gdf is None:
            print("No data found to map.")
            return

        # --- CLIPPING LOGIC ---
        # We clip the S3 shapefiles using the EcoRegions as the mask
        if s3_gdf is not None and eco_gdf is not None:
            print("Clipping S3 layers to EcoRegion boundaries...")
            # We use eco_gdf.unary_union to treat all ecoregions as a single clipping mask
            s3_gdf = gpd.clip(s3_gdf, eco_gdf)
            # Remove any empty geometries resulting from the clip
            s3_gdf = s3_gdf[~s3_gdf.is_empty]

        # 2. Initialize Map (Florida View)
        m = folium.Map(
            location=[27.7, -83.3], 
            zoom_start=7,
            control_scale=True
        )

        # 3. Add EcoRegions Layer
        if eco_gdf is not None:
            eco_layer = folium.FeatureGroup(name="EcoRegions (Boundaries)", show=True)
            folium.GeoJson(
                eco_gdf,
                style_function=lambda x: {
                    'fillColor': 'none', # Keep the ecoregion clear so we can see the clipped shapes inside
                    'color': 'darkgreen', 
                    'weight': 2, 
                    'opacity': 0.7
                },
                tooltip=folium.GeoJsonTooltip(fields=['EcoRegion'] if 'EcoRegion' in eco_gdf.columns else [])
            ).add_to(eco_layer)
            eco_layer.add_to(m)

        # 4. Add Clipped S3 Shapefiles Layer
        if s3_gdf is not None and not s3_gdf.empty:
            s3_layer = folium.FeatureGroup(name="S3 Shapes (Clipped to Ecoregions)", show=True)
            geojson_layer = folium.GeoJson(
                s3_gdf,
                style_function=lambda x: {
                    'fillColor': '#3186cc', 
                    'color': 'black', 
                    'weight': 1, 
                    'fillOpacity': 0.5 # Increased slightly for visibility
                },
                tooltip=folium.GeoJsonTooltip(fields=['filename'], aliases=['File:'])
            ).add_to(s3_layer)
            s3_layer.add_to(m)

            # --- Injection for Point-in-Polygon ---
            m.get_root().header.add_child(Element('<script src="https://unpkg.com/@mapbox/leaflet-pip@latest/leaflet-pip.js"></script>'))
            
            precise_click_js = f"""
            window.addEventListener('load', function() {{
                var mapObj = {m.get_name()};
                var layerObj = {geojson_layer.get_name()};
                
                mapObj.on('click', function(e) {{
                    var res = leafletPip.pointInLayer(e.latlng, layerObj);
                    var content = "<b>Files covering this point: " + res.length + "</b><hr>";
                    
                    if (res.length > 0) {{
                        res.forEach(function(match) {{
                            content += "• " + match.feature.properties.filename + "<br>";
                        }});
                    }} else {{
                        content += "No coverage at this location.";
                    }}
                    
                    L.popup()
                        .setLatLng(e.latlng)
                        .setContent(content)
                        .openOn(mapObj);
                }});
            }});
            """
            m.get_root().script.add_child(Element(precise_click_js))

        # 5. Add Layer Control
        LayerControl(collapsed=False).add_to(m)
        
        m.save(output_file)
        print(f"Interactive map created: {output_file}")