import rasterio
import matplotlib.pyplot as plt
import numpy as np

def plot_improved_histogram(file_path):
    print(f"Loading data from {file_path}...")
    
    with rasterio.open(file_path) as src:
        data = src.read(1)
        nodata_val = src.nodata
        
        # Flatten the array
        flat_data = data.flatten()
        
        # 1. Clean data: Remove NoData and NaNs
        if nodata_val is not None:
            flat_data = flat_data[flat_data != nodata_val]
        flat_data = flat_data[~np.isnan(flat_data)]
        
        # 2. Crucial for AIS: Remove 0s. 
        # (Zeros represent empty ocean which overwhelms the histogram)
        flat_data = flat_data[flat_data > 0]

    print(f"Generating histogram for {len(flat_data)} valid non-zero pixels...")

    plt.figure(figsize=(12, 7))
    
    # 3. Create logarithmically spaced bins for the X-axis
    # We find the min/max and create 100 bins evenly spaced in log-space
    min_val = flat_data.min()
    max_val = flat_data.max()
    log_bins = np.logspace(np.log10(min_val), np.log10(max_val), 100)
    
    # 4. Plot using the new log bins
    plt.hist(flat_data, bins=log_bins, color='seagreen', edgecolor='black', alpha=0.8)
    
    # 5. Set X-axis to log scale, and explicitly ensure Y is linear
    plt.xscale('log')
    plt.yscale('linear')
    
    # Labels and formatting
    plt.title('Distribution of Non-Zero AIS Data (2024 Pilot)\nLogarithmic X-Axis', fontsize=14)
    plt.xlabel('Annual AIS Density (Log Scale)', fontsize=12)
    plt.ylabel('Frequency (Pixel Count)', fontsize=12)
    
    # Improved grid formatting for log X-axis ticking
    plt.grid(axis='x', which='both', alpha=0.3) # Adds gridlines for major & minor log ticks
    plt.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    tif_path = r"C:\Users\aubrey.mccutchan\Documents\AIS_2024_Annual_Density_Pilot.tif"
    plot_improved_histogram(tif_path)