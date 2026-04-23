import pytest
import pathlib

HYDRO_HEALTH_MODULE = pathlib.Path(__file__).parents[2] / 'src'

import sys
sys.path.append(str(HYDRO_HEALTH_MODULE))


from hydro_health.helpers import tools

OUTPUTS = pathlib.Path(__file__).parents[2] / 'outputs'


@pytest.fixture
def victim():
    return tools


@pytest.fixture
def param_lookup():
    return {"eco_regions": tools.Param('ER_1-Texas;'), "output_directory": tools.Param(OUTPUTS), "drawn_polygon": tools.Param('')}


def test_create_raster_vrts(victim):
    ...


def test_get_environment(victim):
    result = victim.get_environment()
    assert result == 'local'


def test_get_ecoregion_folders(victim, param_lookup):
    result = victim.get_ecoregion_folders(param_lookup)
    assert result == ['ER_1']


def test_get_ecoregion_tiles(victim, param_lookup):
    result = victim.get_ecoregion_tiles(param_lookup)
    assert result.shape[0] == 240


def test_get_config_item(victim):
    result = victim.get_config_item('SHARED', 'ECOREGIONS')
    assert result == 'EcoRegions_50m'

    result = victim.get_config_item('TSM', 'DATA_PATH')
    assert result == r'ER_3\original_data_files\tsm_data\nc_files'

    result = victim.get_config_item('TSM', 'DATA_PATH', 'remote')
    assert result == r'\\nos.noaa\ocs\CSDL\Projects\Hydro_Health_Model\HHM2025\working\HHM_Run\ER_3\original_data_files\tsm_data\nc_files'


def test_grid_digital_coast_files(victim):
    ...


def test_make_ecoregion_folders(victim):
    """Already tested with tools.get_ecoregion_folders()"""
    ...


def test_project_raster_wgs84(victim):
    ...


def test_run_vrt_creation(victim):
    ...