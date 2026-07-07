import os
import numpy as np
import rasterio
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.patches as mpatches
import contextily as ctx
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
from pyproj import Transformer

def create_ukc_visualization(region_id, config, base_dir, show_plot=False):
    """
    Generates a UKC map for a specific region with an inset map for a notable port.
    """
    print(f"Processing {region_id} ({config['port_name']})...")
    
    # Construct file paths dynamically based on the region
    tif_path = os.path.join(base_dir, f"{region_id}_UKC_Mosaic_100m.tif")
    output_pdf = os.path.join(base_dir, f"{region_id}_UKC_Map_HighRes.pdf")
    
    # Check if the file exists before attempting to process
    if not os.path.exists(tif_path):
        print(f"  -> File not found: {tif_path}. Skipping {region_id}.")
        return

    # 1. Read the GeoTIFF
    try:
        with rasterio.open(tif_path) as src:
            data = src.read(1)
            nodata_val = src.nodata
            crs = src.crs
            bounds = src.bounds
            extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]
            transform = src.transform
    except Exception as e:
        print(f"  -> Error loading TIFF file for {region_id}: {e}")
        return

    # 2. Handle NoData and calculate valid data mask
    if nodata_val is not None:
        valid_mask = (data != nodata_val) & ~np.isnan(data)
    else:
        valid_mask = ~np.isnan(data)
        
    valid_data = data[valid_mask]
    total_valid = valid_data.size
    
    # Calculate bounding box of only valid data to aggressively crop out empty margins
    rows = np.any(valid_mask, axis=1)
    cols = np.any(valid_mask, axis=0)
    if np.any(rows) and np.any(cols):
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        # Convert pixel indices back to spatial coordinates for the true data bounds
        left_bound, top_bound = transform * (cmin, rmin)
        right_bound, bottom_bound = transform * (cmax + 1, rmax + 1)
        valid_extent = [left_bound, right_bound, bottom_bound, top_bound]
    else:
        valid_extent = extent

    # Hide NoData cells in the plot by setting them to NaN
    plot_data = np.where(valid_mask, data, np.nan)

    # 3. Define Classification Bins and Colors
    bins = [-np.inf, -25.0, -20.0, -15.0, -10.0, -5.0, 0.0, np.inf]
    colors = [
        '#000000',  # Black (<= -25)
        '#bf00ff',  # Purple (-24 to -20)
        '#ff66cc',  # Pink (-19 to -15)
        '#33cc33',  # Green (-14 to -10)
        '#0066ff',  # Blue (-9 to -5)
        '#00ffff',  # Cyan (-4 to 0)
        '#cccccc'   # Grey (> 0)
    ]
    
    cmap = ListedColormap(colors)
    cmap.set_bad(color='none') 
    norm = BoundaryNorm(bins, cmap.N)

    labels = ["<= -25", "-25 to -20", "-20 to -15", "-15 to -10", "-10 to -5", "-5 to 0", "> 0"]

    # 4. Calculate Summary Statistics
    summary_lines = [f"{region_id} Summary (m)", "(% of Valid Cells):", "-"*26]
    for i in range(len(bins) - 1):
        lower = bins[i]
        upper = bins[i+1]
        
        if i == 0:
            count = np.sum(valid_data <= upper)
        elif i == len(bins) - 2:
            count = np.sum(valid_data > lower)
        else:
            count = np.sum((valid_data > lower) & (valid_data <= upper))
            
        pct = (count / total_valid) * 100 if total_valid > 0 else 0
        summary_lines.append(f"{labels[i]:>10}: {pct:5.2f}%")
        
    summary_text = "\n".join(summary_lines)

    # 5. Set up the Main Plot
    fig, ax = plt.subplots(figsize=(12, 10))
    
    im = ax.imshow(plot_data, cmap=cmap, norm=norm, extent=extent, origin='upper', zorder=2, interpolation='none')
    
    try:
        ctx.add_basemap(ax, crs=crs.to_string(), source=ctx.providers.Esri.WorldTopoMap, zorder=1, 
                        attribution="© Esri & contributors", attribution_size=6, zoom=8) 
    except Exception as e:
        print(f"  -> Warning: Could not load basemap for {region_id}. ({e})")

    ax.set_xlim(valid_extent[0], valid_extent[1])
    ax.set_ylim(valid_extent[2], valid_extent[3])
    ax.margins(0) 

    ax.set_title(f"{region_id} UKC (m)", fontsize=18, pad=15)
    ax.set_xticks([])
    ax.set_yticks([])

    # 6. Create Inset / Breakout Box for the specific Region
    wgs84_bounds = config["bounds"] # [min_lon, max_lon, min_lat, max_lat]
    
    # Transform coordinates to the TIFF's CRS
    transformer = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    tb_xmin, tb_ymin = transformer.transform(wgs84_bounds[0], wgs84_bounds[2])
    tb_xmax, tb_ymax = transformer.transform(wgs84_bounds[1], wgs84_bounds[3])

    # Extract inset location and size based on the config
    loc_str = config.get("inset_loc", "lower left")
    inset_size = config.get("inset_size", "40%") # Use size from config, default to 40%
    
    # We want the inset to always stay completely inside the main axes.
    bbox_anchor = (0, 0, 1, 1)

    # Create inset axes dynamically sized
    axins = inset_axes(ax, width=inset_size, height=inset_size, loc=loc_str, 
                       bbox_to_anchor=bbox_anchor, bbox_transform=ax.transAxes, borderpad=0)
    
    # CRITICAL FIX: Because map data uses aspect='equal', the inset box often shrinks 
    # to maintain the geographic shape (e.g., Mobile Bay is tall, so the box width shrinks). 
    # By default, Matplotlib centers ('C') this shrunken box, causing gaps at the edges. 
    # This mapping forces the box to anchor flush into the designated corner instead.
    anchor_mapping = {
        "lower left": "SW",
        "lower right": "SE",
        "upper left": "NW",
        "upper right": "NE"
    }
    axins.set_anchor(anchor_mapping.get(loc_str, 'C'))
    
    axins.imshow(plot_data, cmap=cmap, norm=norm, extent=extent, origin='upper', zorder=2, interpolation='none')
    
    try:
        ctx.add_basemap(axins, crs=crs.to_string(), source=ctx.providers.Esri.WorldTopoMap, zorder=1, 
                        attribution=False, zoom=10)
    except:
        pass 

    # Set inset limits bounds
    axins.set_xlim(tb_xmin, tb_xmax)
    axins.set_ylim(tb_ymin, tb_ymax)
    axins.set_xticks([]) 
    axins.set_yticks([])
    axins.set_title(config["port_name"], fontsize=12, backgroundcolor="white", alpha=0.8)
    
    for spine in axins.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(0.5) # Made thinner

    # Determine mark_inset corners explicitly from the config
    # loc1, loc2 map to corners: 1=top right, 2=top left, 3=bottom left, 4=bottom right
    loc1, loc2 = config.get("mark_locs", (2, 4))
    
    # Draw the connecting breakout box and capture the drawn elements
    patch, p1, p2 = mark_inset(ax, axins, loc1=loc1, loc2=loc2, fc="none", ec="black", lw=0.5, zorder=4) # Made thinner

    # CRITICAL FIX: Matplotlib doesn't clip inset lines by default. 
    # If we only set clip_on=True, it defaults to clipping them to the inset axes, 
    # causing them to disappear. We must explicitly set the clip box to the main axes (ax.bbox).
    patch.set_clip_box(ax.bbox)
    patch.set_clip_on(True)
    p1.set_clip_box(ax.bbox)
    p1.set_clip_on(True)
    p2.set_clip_box(ax.bbox)
    p2.set_clip_on(True)

    # 7. Create the Legend
    legend_patches = [mpatches.Patch(facecolor=colors[i], edgecolor='none', label=labels[i]) for i in range(len(colors))]
    ax.legend(handles=legend_patches, loc='upper left', bbox_to_anchor=(1.02, 1), 
              title="UKC (m)", fontsize=10, title_fontsize=12, framealpha=0.9, edgecolor='black')

    # 8. Add Summary Text
    ax.text(1.02, 0.55, summary_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top', horizontalalignment='left',
            multialignment='right',
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="black", alpha=0.9),
            zorder=5, family='monospace')

    plt.subplots_adjust(left=0.01, bottom=0.02, top=0.92, right=0.68)
    plt.savefig(output_pdf, format='pdf', dpi=600, bbox_inches='tight')
    print(f"  -> Successfully saved high-resolution PDF to: {output_pdf}")
    
    if show_plot:
        plt.show()
    else:
        plt.close(fig)

if __name__ == "__main__":
    # Define the base directory where all the TIFF files are located
    base_directory = r"C:\Users\aubrey.mccutchan\Documents"
    
    # Configuration dictionary for all ERs. 
    #   - `bounds`: [min_lon, max_lon, min_lat, max_lat]
    #   - `inset_size`: Adjust percentage string (e.g. "30%", "45%") to shrink/grow the breakout box.
    #   - `mark_locs`: (Corner1, Corner2) where 1=TR, 2=TL, 3=BL, 4=BR. Use this to stop lines from crossing your map weirdly.
    regions_config = {
        "ER_1": {
            "port_name": "Galveston Bay",
            "bounds": [-95.2, -94.6, 29.1, 29.8],
            "inset_loc": "lower right",
            "inset_size": "40%",
            "mark_locs": (1, 2) # Top Right, Top Left
        },
        "ER_2": {
            "port_name": "Mississippi Sound",
            "bounds": [-89.25, -88.45, 30.1, 30.5],
            "inset_loc": "lower right", 
            "inset_size": "40%", 
            "mark_locs": (1, 2) # Top Right, Top Left
        },
        "ER_3": {
            "port_name": "Tampa Bay",
            "bounds": [-82.90, -82.25, 27.45, 28.05],
            "inset_loc": "lower left",
            "inset_size": "40%",
            "mark_locs": (1, 2) # Top Right, Top Left
        },
        "ER_4": {
            "port_name": "PortMiami Area",
            "bounds": [-80.22, -80.02, 25.55, 25.85], # Shifted down and centered horizontally
            "inset_loc": "lower right",
            "inset_size": "40%",
            "mark_locs": (1, 2) # Top Right, Top Left
        },
        "ER_5": {
            "port_name": "Port of Wilmington",
            "bounds": [-78.10, -77.60, 33.70, 34.15], # Shifted down and expanded right
            "inset_loc": "lower right",
            "inset_size": "40%",
            "mark_locs": (1, 2) # Top Right, Top Left
        },
        "ER_6": {
            "port_name": "New York Harbor",
            "bounds": [-74.25, -73.75, 40.30, 40.75], # Shifted south to focus more on Lower Bay / channels
            "inset_loc": "lower right",
            "inset_size": "40%",
            "mark_locs": (1, 2) # Top Right, Top Left
        }
    }

    # Loop through all regions in the dictionary and generate maps
    for er_name, config_data in regions_config.items():
        create_ukc_visualization(
            region_id=er_name, 
            config=config_data, 
            base_dir=base_directory, 
            show_plot=False
        )