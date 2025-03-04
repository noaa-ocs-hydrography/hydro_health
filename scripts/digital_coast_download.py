import pathlib


def process():
    bbox_string = '-9279455.233820397,3275785.284189513,-9196597.495159267,3368732.710584287'  # BL to TR
    bbox_url = f'https://coast.noaa.gov/dataviewer/#/lidar/search/{bbox_string}'
    ags_rest_url = 'https://maps.coast.noaa.gov/arcgis/rest/services/DAV/ElevationFootprints/MapServer/0/query?returnGeometry=false&f=json&where=1%3D1&outfields=%2A&spatialRel=esriSpatialRelIntersects&geometry=-9288933.42532776%2C3152874.5427069496%2C-9186202.059312481%2C3269058.8257004176'

    # the ags rest URL returns a JSON of datasets.  Need to try to anonymously list files in bucket after getting list of features from REST: noaa-nos-coastal-lidar-pds.s3.amazonaws.com
    # ExternalProviderLink then links[0]
if __name__ == "__main__":
    process()