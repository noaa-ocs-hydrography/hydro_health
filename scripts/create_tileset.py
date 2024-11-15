import os
from osgeo import osr, ogr, gdal
import numpy as np
import pathlib

gdal.SetConfigOption('OGR_SQLITE_PRAGMA', 'journal_mode=MEMORY')


OUTPUTS = pathlib.Path(__file__).parents[1] / 'outputs'


def convert_base(charset: str, input: int, minimum: int) -> str:
    """
    Convert integer to new base system using the given symbols with a
    minimum length filled using leading characters of the lowest value in the
    given charset.

    Parameters
    ----------
    charset : str
        length of this str will be the new base system and characters
        given will be the symbols used.
    input : int
        integer to convert.
    minimum : int
        returned output will be adjusted to this desired length using
        leading characters of the lowest value in charset.

    Returns
    -------
    str
        converted value in given system.
    """
    res = ""
    while input:
        res += charset[input % len(charset)]
        input //= len(charset)
    return (res[::-1] or charset[0]).rjust(minimum, charset[0])


def global_region_tileset(tileset_name, sizes) -> str:
    """
    Parameters
    ----------

    Returns
    -------
    """
    for name, size_str in sizes.items():
        charset="BCDFGHJKLMNPQRSTVWXZ"
        name = convert_base(charset, name, 2)
        roundnum = len(size_str.split('.')[1])
        size = float(size_str)
        location = str(OUTPUTS / f"{tileset_name}_{name}.gpkg")
        ds = ogr.GetDriverByName('GPKG').CreateDataSource(location)
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(4326)
        layer = ds.CreateLayer(f'Tileset_{name}_{size_str.replace('.', 'pt')}', srs, ogr.wkbMultiPolygon)
        layer.CreateFields([ogr.FieldDefn("Tilename", ogr.OFTString), ogr.FieldDefn("Tileset", ogr.OFTString),
                            ogr.FieldDefn("UTM_Zone", ogr.OFTInteger), ogr.FieldDefn("Hemisphere", ogr.OFTString),
                            ogr.FieldDefn("Resolution", ogr.OFTString)])
        layer_defn = layer.GetLayerDefn()
        layer.StartTransaction()
        y = round(-90+size, roundnum)
        y_count = 0
        while y <= 90:
            ns = "N"
            if y <= 0:
                ns = "S"
            x = -180
            x_count = 0
            while x < 180:
                current_utm = "{:02d}".format(int(np.ceil((180+x+.00000001)/6)))
                ring = ogr.Geometry(ogr.wkbLinearRing)
                ring.AddPoint_2D(x, y)
                ring.AddPoint_2D(round(x+size,roundnum), y)
                ring.AddPoint_2D(round(x+size,roundnum), round(y-size,roundnum))
                ring.AddPoint_2D(x, round(y-size,roundnum))
                ring.AddPoint_2D(x, y)
                poly = ogr.Geometry(ogr.wkbPolygon)
                poly.AddGeometry(ring)
                multipoly = ogr.Geometry(ogr.wkbMultiPolygon)
                multipoly.AddGeometry(poly)
                feat = ogr.Feature(layer_defn)
                feat.SetGeometry(multipoly)
                charset="2456789BCDFGHJKLMNPQRSTVWXZ"
                x_rep = convert_base(charset, x_count, 3)
                y_rep = convert_base(charset, y_count, 3)
                feat.SetField('Tilename', f'{name}{x_rep}{y_rep}')
                feat.SetField('Tileset', tileset_name)
                feat.SetField('UTM_Zone', current_utm)
                feat.SetField('Hemisphere', ns)
                layer.CreateFeature(feat)
                x = round(x+size, roundnum)
                x_count += 1
            y = round(y+size, roundnum)
            y_count += 1
            print(x, y)
        layer.CommitTransaction()


# tileset_sizes is a dict 
# the key is an integer which determines the tilename prefix ("BB", "BC", etc) 
# the value is the size as a string and necessarily includes a decimal

# tileset_name = "Global"
# tileset_sizes = {0:'6.', 1:'1.2', 2:'.6', 3:'.3', 4:'.15', 5:'.075'}
# global_region_tileset(tileset_name, tileset_sizes)

# tileset_name = "Global"
# tileset_sizes = {6:'.0375', 7:'.01875'}
# output_directory = "/folder/Tilesets/"
# global_region_tileset(tileset_name, tileset_sizes, output_directory)


# Testing
# tileset_name = "Global"
# tileset_sizes = {1:'1.2', 2:'.6', 3:'.3'}
# global_region_tileset(tileset_name, tileset_sizes)

tileset_name = "Global"
tileset_sizes = {0:'6.', 4:'.15', 5:'.075'}
global_region_tileset(tileset_name, tileset_sizes)