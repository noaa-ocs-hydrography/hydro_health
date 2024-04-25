import yaml
import pathlib


INPUTS = pathlib.Path(__file__).parents[1] / 'inputs'


class Engine:
    def get_config_item(item:str) -> str:
        """Load config and return speciific key"""

        with open(str(INPUTS / 'config.yaml'), 'r') as lookup:
            config = yaml.safe_load(lookup)
        return config[item]
