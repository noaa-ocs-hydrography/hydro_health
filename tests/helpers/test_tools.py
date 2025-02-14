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
