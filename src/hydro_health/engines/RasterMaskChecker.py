import boto3
import geopandas as gpd
import pandas as pd
import folium
from folium import Element, LayerControl
import os
import tempfile
import pathlib
import yaml

# Assuming these exist in your environment
from hydro_health.helpers.tools import get_config_item, get_approved_providers

INPUTS = pathlib.Path(__file__).parents[3] / 'inputs'
OUTPUTS = pathlib.Path(__file__).parents[3] / 'outputs'

class RasterMaskChecker:
    def __init__(self):
        self.s3 = boto3.client('s3')
        self.bucket = get_config_item('SHARED', 'OUTPUT_BUCKET')
        self.prefix = 'ER_3/model_variables/Prediction/raw/DigitalCoast'

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
                if ext == '.shp': 
                    local_shp_path = local_file
            except Exception:
                continue 
        return local_shp_path

    def load_s3_shapefiles(self, ecoregion='ER_3', simplify_tolerance=0.0001):
        """Finds shapefiles, filters by approved providers, dissolves, and returns GeoDataFrame."""
        # Get list of approved providers in lowercase for case-insensitive comparison
        approved_providers = [p.lower() for p in get_approved_providers(ecoregion)]
        paginator = self.s3.get_paginator('list_objects_v2')
        all_gdfs = []

        for page in paginator.paginate(Bucket=self.bucket, Prefix=self.prefix):
            for obj in page.get('Contents', []):
                key = obj['Key']
                if key.endswith('.shp') and 'tileindex' in key.lower() and 'NCEI' not in key:
                    
                    # Extract provider folder name from the S3 key
                    provider_folder = "Unknown"
                    parts = key.split('/')
                    if 'DigitalCoast' in parts:
                        dc_index = parts.index('DigitalCoast')
                        if dc_index + 1 < len(parts):
                            provider_folder = parts[dc_index + 1]

                    # Filter out providers that are not in the approved list
                    provider_end = '_'.join(provider_folder.lower().split('_')[1:])
                    
                    if provider_end not in approved_providers:
                        print(f' - skipping unapproved provider {provider_end}')
                        continue

                    path = self._download_shapefile_set(key)
                    if path:
                        gdf = gpd.read_file(path).to_crs(epsg=4326)
                        
                        # Dissolve all features into a single geometry per shapefile
                        gdf_dissolved = gdf.dissolve()
                        
                        gdf_dissolved['geometry'] = gdf_dissolved.simplify(
                            tolerance=simplify_tolerance, preserve_topology=True
                        )
                        gdf_dissolved['filename'] = os.path.basename(key)
                        gdf_dissolved['provider_folder'] = provider_folder
                        all_gdfs.append(gdf_dissolved[['filename', 'provider_folder', 'geometry']])
        
        return pd.concat(all_gdfs, ignore_index=True) if all_gdfs else None

    def load_ecoregions(self, gpkg_path, layer_name='Enhanced_EcoRegions_50m'):
        """Loads the specific EcoRegions layer from the local GeoPackage."""
        if not os.path.exists(gpkg_path):
            print(f"Warning: GeoPackage not found at {gpkg_path}")
            return None
        eco_gdf = gpd.read_file(gpkg_path, layer=layer_name).to_crs(epsg=4326)
        return eco_gdf

    def create_interactive_map(self, ecoregion='ER_3', output_file=None):
        if output_file is None:
            output_file = pathlib.Path(OUTPUTS) / 'overlap_check.html'

        # 1. Load Data (Passing ecoregion to filter approved providers)
        s3_gdf = self.load_s3_shapefiles(ecoregion=ecoregion)
        eco_gdf = self.load_ecoregions(str(INPUTS / get_config_item('SHARED', 'MASTER_GRIDS')))

        if s3_gdf is None and eco_gdf is None:
            print("No data found to map.")
            return

        # --- CLIPPING & CLEANING LOGIC ---
        if s3_gdf is not None and eco_gdf is not None:
            print("Clipping S3 layers to EcoRegion boundaries...")
            s3_gdf = gpd.clip(s3_gdf, eco_gdf)
            s3_gdf = s3_gdf[~s3_gdf.is_empty & s3_gdf.geometry.notnull()]
            s3_gdf = s3_gdf.reset_index(drop=True)

        # 2. Initialize Map (Florida View) with full dimensions
        m = folium.Map(
            location=[27.7, -83.3], 
            zoom_start=7,
            control_scale=True,
            width='100%',
            height='100%'
        )

        # Force body/html layout CSS fix and silence favicon request
        css_fix = """
        <style>
            html, body {width: 100%; height: 100%; margin: 0; padding: 0;}
            .folium-map {width: 100%; height: 100vh !important;}
        </style>
        <link rel="shortcut icon" href="data:image/x-icon;," type="image/x-icon">
        """
        m.get_root().header.add_child(Element(css_fix))

        # 3. Add EcoRegions Layer
        if eco_gdf is not None:
            eco_layer = folium.FeatureGroup(name="EcoRegions (Boundaries)", show=True)
            folium.GeoJson(
                eco_gdf,
                style_function=lambda x: {
                    'fillColor': 'none',
                    'color': 'darkgreen', 
                    'weight': 2, 
                    'opacity': 0.7
                },
                tooltip=folium.GeoJsonTooltip(fields=['EcoRegion'] if 'EcoRegion' in eco_gdf.columns else [])
            ).add_to(eco_layer)
            eco_layer.add_to(m)

        # 4. Add Individual S3 Shapefile Layers for Toggling
        layer_info_map = []
        if s3_gdf is not None and not s3_gdf.empty:
            for idx in range(len(s3_gdf)):
                single_gdf = s3_gdf.iloc[[idx]].copy()
                filename = str(single_gdf.iloc[0]['filename'])
                provider_folder = str(single_gdf.iloc[0]['provider_folder'])
                
                # Format layer pane label using Provider Folder + Filename
                layer_label = f"{provider_folder}"
                file_layer = folium.FeatureGroup(name=layer_label, show=True)
                
                geojson_feature = folium.GeoJson(
                    single_gdf,
                    style_function=lambda x: {
                        'fillColor': '#3186cc', 
                        'color': 'black', 
                        'weight': 1, 
                        'fillOpacity': 0.5
                    },
                    tooltip=folium.GeoJsonTooltip(
                        fields=['provider_folder', 'filename'], 
                        aliases=['Provider:', 'File:']
                    )
                ).add_to(file_layer)
                
                file_layer.add_to(m)
                
                # Store layer mapping for click events
                layer_info_map.append(f"{{layer: {geojson_feature.get_name()}, name: '{layer_label}'}}")
            
            # Pure Native Leaflet Spatial Bounds Click Listener
            map_name = m.get_name()
            js_layers_array = ", ".join(layer_info_map)
            
            native_click_js = f"""
            window.addEventListener('load', function() {{
                var mapObj = {map_name};
                var targetLayers = [{js_layers_array}];
                
                mapObj.on('click', function(e) {{
                    var matches = [];
                    
                    targetLayers.forEach(function(item) {{
                        if (mapObj.hasLayer(item.layer)) {{
                            item.layer.eachLayer(function(subLayer) {{
                                if (subLayer.getBounds && subLayer.getBounds().contains(e.latlng)) {{
                                    matches.push(item.name);
                                }}
                            }});
                        }}
                    }});
                    
                    var content = "<b>Active files covering this point: " + matches.length + "</b><hr>";
                    if (matches.length > 0) {{
                        matches.forEach(function(fname) {{
                            content += "• " + fname + "<br>";
                        }});
                    }} else {{
                        content += "No active coverage at this location.";
                    }}
                    
                    L.popup()
                        .setLatLng(e.latlng)
                        .setContent(content)
                        .openOn(mapObj);
                }});
            }});
            """
            m.get_root().script.add_child(Element(native_click_js))

        # 5. Add Layer Control
        LayerControl(collapsed=False).add_to(m)
        
        m.save(output_file)
        print(f"Interactive map created: {output_file}")