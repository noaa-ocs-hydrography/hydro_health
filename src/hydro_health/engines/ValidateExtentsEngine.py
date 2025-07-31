import pathlib

from hydro_health.helpers.tools import get_config_item, get_environment


INPUTS = pathlib.Path(__file__).parents[3] / 'inputs'
OUTPUTS = pathlib.Path(__file__).parents[3] / 'outputs'


class ValidateExtentsEngine:
    """Class for creating year-pair extent polygons"""

    def __init__(self) -> None:
        self.digital_coast_folder = (
            OUTPUTS / pathlib.Path(get_config_item("DIGITALCOAST", "DATA_PATH"))
            if get_environment() == "local"
            else pathlib.Path(get_config_item("SHARED", "OUTPUT_FOLDER")) / pathlib.Path(get_config_item("DIGITALCOAST", "DATA_PATH"))
        )

    def get_year_pairs(self) -> None:
        """Build year pairs from datasets"""

        provider_folders = [folder for folder in self.digital_coast_folder.iterdir() if folder.is_dir() and folder.stem != 'tiled']
        years = [folder.stem[-4:] for folder in provider_folders]
        #  TODO need to keep track of folder.  Maybe have duplicate years if year alone
        print(sorted(years))
        # print(provider_folders)
        sorted_years = sorted(set(years))
        print(sorted_years, len(sorted_years))

    def run(self) -> None:
        # get year pairs of digital coast
        self.get_year_pairs()
        return