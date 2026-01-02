from pathlib import Path

# Define the path to your directory
directory_path = Path(r"N:\CSDL\Projects\Hydro_Health_Model\HHM2025\working\HHM_Run\ER_3\model_variables\Prediction\raw\DigitalCoast") # Replace with your actual path

# Use a list comprehension to get all folders and .vrt files
matching_items = [item for item in directory_path.iterdir() 
                  if item.is_dir() or item.suffix == '.vrt']

# Print the full path of each matching item
for item in matching_items:
    print(item)