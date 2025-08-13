import pathlib

from hydro_health.helpers.tools import get_config_item, get_environment


INPUTS = pathlib.Path(__file__).parents[3] / 'inputs'
OUTPUTS = pathlib.Path(__file__).parents[3] / 'outputs'


class ValidateExtentsEngine:
    """Class for creating year-pair extent polygons"""

    def __init__(self) -> None:
        # self.output_folder = (
        #     OUTPUTS
        #     if get_environment() == "local"
        #     else pathlib.Path(get_config_item("SHARED", "OUTPUT_FOLDER"))
        # )

        # TODO force remote for testing
        self.output_folder = pathlib.Path(get_config_item("SHARED", "OUTPUT_FOLDER", 'remote'))
        # TODO make subfolders and if/else with empty string for local or change remote folders
        self.subfolders = pathlib.Path(get_config_item("DIGITALCOAST", "SUBFOLDERS", 'remote'))

    def get_year_pairs(self) -> None:
        """Build year pairs from datasets"""

        for ecoregion in [folder for folder in self.output_folder.iterdir() if folder.is_dir() and 'ER' in folder.stem]:
            print(ecoregion.stem)
            digital_coast_folder = ecoregion / self.subfolders / 'DigitalCoast'
            if digital_coast_folder.exists():
                provider_folders = [folder for folder in digital_coast_folder.iterdir() if folder.is_dir() and folder.stem != 'tiled']
                years = [folder.stem[-4:] for folder in provider_folders]
                #  TODO need to keep track of folder.  Maybe have duplicate years if year alone
                print('Years found:', sorted(years), len(years))
                sorted_years = sorted(years)
                # sorted_years = sorted(set(years))
                # print(sorted_years, len(sorted_years))
                year_pairs = []
                for i in range(len(sorted_years) - 1):
                    start, end = sorted_years[i], sorted_years[i+1]
                    if start == end:
                        print('wat?! Multiple data for year:', start)
                        # TODO bundle same year data for model?
                    else:
                        year_pairs.append(f'{start}-{end}')
                print(year_pairs)


    def run(self) -> None:
        # get year pairs of digital coast
        self.get_year_pairs()
        return