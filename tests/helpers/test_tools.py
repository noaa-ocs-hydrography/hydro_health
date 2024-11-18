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
    # # pytest -s C:\Users\Stephen.Patterson\Data\Repos\hydro_health\tests\helpers\test_tools.py
    param_lookup={"coastal_states": Param('California;Florida;North Carolina')}
    result = victim.get_state_tiles(param_lookup)
