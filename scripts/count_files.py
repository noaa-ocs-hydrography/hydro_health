import os
import cProfile
import pstats
import io

def get_tif_stats(directory_path):
    """Counts .tif/.tiff files and calculates their total size in bytes.

    param str directory_path: The absolute or relative path to the directory.
    return: A tuple containing the total count (int) and total size (int) in bytes.
    """
    tif_count = 0
    total_size_bytes = 0
    
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if file.lower().endswith(('.tif', '.tiff')):
                file_path = os.path.join(root, file)
                try:
                    # Add file size to total and increment count
                    total_size_bytes += os.path.getsize(file_path)
                    tif_count += 1
                except os.error:
                    # Skip files that cannot be accessed (e.g., permissions)
                    pass 
                    
    return tif_count, total_size_bytes

def format_size(size_bytes):
    """Converts bytes to a human-readable format.

    param int size_bytes: Size in bytes.
    return: A formatted string representing the size (e.g., "10.50 MB").
    """
    if size_bytes == 0:
        return "0 B"
    size_name = ("B", "KB", "MB", "GB", "TB", "PB")
    i = 0
    while size_bytes >= 1024 and i < len(size_name) - 1:
        size_bytes /= 1024.0
        i += 1
    return f"{size_bytes:.2f} {size_name[i]}"

def run_profiler(directory_path):
    """Runs the file stats function under cProfile and prints results.

    param str directory_path: The directory to analyze.
    return: None.
    """
    profiler = cProfile.Profile()
    
    # Run the target function under the profiler
    profiler.enable()
    total_files, total_bytes = get_tif_stats(directory_path)
    profiler.disable()
    
    # Print the original results
    readable_size = format_size(total_bytes)
    print(f"Total .tif/.tiff files found in '{directory_path}': {total_files}")
    print(f"Total memory usage: {readable_size} ({total_bytes} bytes)")
    
    print("\n--- cProfile Stats ---")
    
    # Sort and print profiler stats
    s = io.StringIO()
    # Sort by 'cumulative' time spent in the function
    ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative') 
    ps.print_stats(15) # Print top 15 lines
    
    print(s.getvalue())

if __name__ == "__main__":
    # Specify the directory you want to search
    search_directory = r"N:\CSDL\Projects\Hydro_Health_Model\HHM2025\working\HHM_Run\ER_3\model_variables\Prediction\processed\terrain_digitalcoast_outputs\filled_tifs"
    
    run_profiler(search_directory)