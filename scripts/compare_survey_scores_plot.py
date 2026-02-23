import rasterio
from rasterio.plot import show
import matplotlib.pyplot as plt
import contextily as cx
from matplotlib.colors import ListedColormap, BoundaryNorm
import numpy as np

# File paths updated to the new requested locations
tif_1_0 = r"N:\CSDL\Projects\Hydro_Health_Model\HHM2025\working\Post_processing_logic\Pilot_Model_Outputs\ISS_values_2018.tif"
tif_2_0 = r"N:\CSDL\Projects\Hydro_Health_Model\HHM2025\working\Post_processing_logic\Pilot_Model_Outputs\ISS_values_2026.tif"

# Define ranges and colors based on pixel values
# Adjust levels to match the range of your actual CATZOC scores
levels = [0, 20, 40, 60, 80, 100] 
colors = ['#d7191c', '#fdae61', '#ffffbf', '#abdda4', '#2b83ba'] # Red to Blue

# Create colormap and normalization
cmap = ListedColormap(colors)
norm = BoundaryNorm(levels, ncolors=len(colors))

def get_plot_data(path):
    """Opens a tif and returns the data, transform, and crs."""
    with rasterio.open(path) as src:
        data = src.read(1).astype(float)
        # Handle nodata specifically so it doesn't plot as a color
        nodata = src.nodata if src.nodata is not None else -9999
        data[data == nodata] = np.nan
        
        # Get the spatial extent (left, right, bottom, top) for imshow
        # This replaces the broken rasterio.plot.get_data_window call
        left, bottom, right, top = src.bounds
        extent = [left, right, bottom, top]
        
        return data, extent, src.crs

# Load data
data1, extent1, crs1 = get_plot_data(tif_1_0)
data2, extent2, crs2 = get_plot_data(tif_2_0)

# Create side-by-side plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10), sharex=True, sharey=True)

def setup_map(ax, data, extent, crs, title):
    # Plot the data using the calculated extent
    im = ax.imshow(data, extent=extent, cmap=cmap, norm=norm, alpha=0.8, zorder=2)
    
    # Add the basemap (zorder 1 puts it behind the data)
    try:
        cx.add_basemap(ax, crs=crs.to_string(), 
                       source=cx.providers.OpenStreetMap.Mapnik, 
                       zorder=1)
    except Exception as e:
        print(f"Could not load basemap for {title}: {e}")

    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.set_axis_off()
    return im

# Execute plotting
im1 = setup_map(ax1, data1, extent1, crs1, "HHM 1.0 ISS")
im2 = setup_map(ax2, data2, extent2, crs2, "HHM 2.0 ISS")

# Add a combined colorbar for pixel values
cbar_ax = fig.add_axes([0.92, 0.25, 0.015, 0.5]) # [left, bottom, width, height]
cbar = fig.colorbar(im1, cax=cbar_ax, spacing='proportional')
cbar.set_label('CATZOC Score (Pixel Value)', fontsize=12, labelpad=15)

# Adjust layout to prevent overlap
plt.subplots_adjust(wspace=0.05, right=0.9)

print("Plotting complete. Displaying window...")
plt.show()