import os
import pathlib
import re
import yaml

INPUTS = pathlib.Path(__file__).parents[1] / 'inputs' / 'lookups'

def generate_ecoregion_yaml(ecoregion_number, base_path, output_filename) -> None:
    """Generates a YAML configuration file for a specific ecoregion.
        param int ecoregion_number: The number of the EcoRegion (e.g., 3).
        param str base_path: The absolute path to the directory containing data folders.
        param str output_filename: The name of the output YAML configuration file.
        return: None.
    """

    section_name = f'EcoRegion-{ecoregion_number}'
    config_data = {section_name: {}}

    try:
        all_dirs = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
    except FileNotFoundError:
        print(f"Error: The directory '{base_path}' was not found.")
        return

    def extract_year(dir_name):
        """Extracts the year from a directory name string.
            param str dir_name: The string containing the directory name.
            return: The extracted four-digit year as an integer, or 0 if not found.
        """
        match = re.search(r'\d{4}', dir_name)
        return int(match.group(0)) if match else 0

    sorted_dirs = sorted(all_dirs, key=extract_year)

    for dir_name in sorted_dirs:
        metadata_path = os.path.join(base_path, dir_name, 'metadata.txt')
        date_str = 'Date not found'

        try:
            with open(metadata_path, 'r') as f:
                content = f.read()
                dates_found = re.findall(r'\d{4}-\d{2}-\d{2}', content)
                if dates_found:
                    date_str = ', '.join(dates_found)
        except FileNotFoundError:
            date_str = 'metadata.txt not found'
        except Exception as e:
            date_str = f"Error reading metadata: {e}"

        config_data[section_name][dir_name] = {
            'use': False,
            'date': date_str,
            'year': None
        }

    try:
        with open(output_filename, 'w') as yamlfile:
            yaml.dump(config_data, yamlfile, default_flow_style=False, sort_keys=False)
        print(f"Successfully generated YAML configuration file: '{output_filename}'")
    except IOError as e:
        print(f"Error: Could not write to file '{output_filename}'. Reason: {e}")

if __name__ == '__main__':
    ECOREGION_ID = 3
    
    DATA_DIRECTORY = r'N:\CSDL\Projects\Hydro_Health_Model\HHM2025\working\HHM_Run\ER_3\model_variables\Prediction\raw\DigitalCoast'

    
    OUTPUT_FILE = os.path.join(INPUTS, f'ER_{ECOREGION_ID}_lidar_data_config_template.yaml')

    generate_ecoregion_yaml(
        ecoregion_number=ECOREGION_ID,
        base_path=DATA_DIRECTORY,
        output_filename=OUTPUT_FILE
    )

