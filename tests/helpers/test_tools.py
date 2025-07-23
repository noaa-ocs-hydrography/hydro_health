import pytest
import pathlib

HYDRO_HEALTH_MODULE = pathlib.Path(__file__).parents[1]

import sys
sys.path.append(str(HYDRO_HEALTH_MODULE))


from hydro_health.helpers import tools


class Param:
    def __init__(self, path):
        self.path = path

    @property
    def valueAsText(self):
        return self.path


@pytest.fixture
def victim():
    return tools


def test_get_state_tiles(victim):
    param_lookup={"coastal_states": Param('California;North Carolina')}
    result = victim.get_state_tiles(param_lookup)
    assert result.shape[0] == 576
    assert 'STATE_NAME' in result.columns.tolist()
    states = result['STATE_NAME'].unique()
    assert 'California' in states
    assert 'North Carolina' in states


def test_get_ecoregion_tiles(victim):
    param_lookup={"eco_regions": Param('ER_1-Texas;')}
    result = victim.get_ecoregion_tiles(param_lookup)
    assert result.shape[0] == 246


def test_get_config_item(victim):
    result = victim.get_config_item('SHARED', 'ECOREGIONS')
    assert result == 'EcoRegions_50m'

    result = victim.get_config_item('TSM', 'DATA_PATH')
    assert result == r'HHM_Run\ER_3\original_data_files\tsm_data\nc_files'

    result = victim.get_config_item('TSM', 'DATA_PATH', 'remote')
    assert result == r'\\nos.noaa\ocs\CSDL\Projects\Hydro_Health_Model\HHM2025\working\HHM_Run\ER_3\original_data_files\tsm_data\nc_files'