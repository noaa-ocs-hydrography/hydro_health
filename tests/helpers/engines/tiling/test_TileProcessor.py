import pytest
import pathlib
import multiprocessing as mp

HYDRO_HEALTH_MODULE = pathlib.Path(__file__).parents[1]

import sys
sys.path.append(str(HYDRO_HEALTH_MODULE))


from hydro_health.engines.BlueTopoEngine import BlueTopoEngine


@pytest.fixture
def victim():
    return BlueTopoEngine()


@pytest.mark.skip(reason="Don't want to download a tile")
def test_download_nbs_tile(victim):
    ...


def test_get_bucket(victim):
    result = victim.get_bucket()
    assert str(type(result)) == "<class 'boto3.resources.factory.s3.Bucket'>"
    assert result.name == 'noaa-ocs-nationalbathymetry-pds'


@pytest.mark.skip(reason="Don't want to download a tile")
def test_process_tile(victim):
    ...

    
@pytest.mark.skip(reason="Don't want to download a tile")
def test_process(victim):
    ...