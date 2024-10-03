
import numpy as np
import pathlib

from osgeo import ogr, osr


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



def global_region_tileset(index: int, size: str) -> str:
    """
    Generate a global tilescheme.

    Parameters
    ----------
    index : int
        index of tileset to determine tilescheme name.
    size : str
        length of the side of an individual tile in degrees.

    Returns
    -------
    location : str
        gdal memory filepath to global tilescheme.
    """
    charset = "BCDFGHJKLMNPQRSTVWXZ"
    name = convert_base(charset, index, 2)
    roundnum = len(size.split(".")[1])
    size = float(size)
    location = str(OUTPUTS / "global_tileset.gpkg")
    ds = ogr.GetDriverByName("GPKG").CreateDataSource(location)
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    layer = ds.CreateLayer("global_tileset", srs, ogr.wkbMultiPolygon)
    layer.CreateFields(
        [
            ogr.FieldDefn("Region", ogr.OFTString),
            ogr.FieldDefn("UTM_Zone", ogr.OFTInteger),
            ogr.FieldDefn("Hemisphere", ogr.OFTString),
        ]
    )
    layer_defn = layer.GetLayerDefn()
    layer.StartTransaction()
    y = round(-90 + size, roundnum)
    y_count = 0
    while y <= 90:
        ns = "N"
        if y <= 0:
            ns = "S"
        x = -180
        x_count = 0
        while x < 180:
            current_utm = "{:02d}".format(int(np.ceil((180 + x + 0.00000001) / 6)))
            ring = ogr.Geometry(ogr.wkbLinearRing)
            ring.AddPoint_2D(x, y)
            ring.AddPoint_2D(round(x + size, roundnum), y)
            ring.AddPoint_2D(round(x + size, roundnum), round(y - size, roundnum))
            ring.AddPoint_2D(x, round(y - size, roundnum))
            ring.AddPoint_2D(x, y)
            poly = ogr.Geometry(ogr.wkbPolygon)
            poly.AddGeometry(ring)
            poly = poly.Buffer(-0.002)
            multipoly = ogr.Geometry(ogr.wkbMultiPolygon)
            multipoly.AddGeometry(poly)
            feat = ogr.Feature(layer_defn)
            feat.SetGeometry(multipoly)
            charset = "2456789BCDFGHJKLMNPQRSTVWXZ"
            x_rep = convert_base(charset, x_count, 3)
            y_rep = convert_base(charset, y_count, 3)
            feat.SetField("Region", f"{name}{x_rep}{y_rep}")
            feat.SetField("UTM_Zone", current_utm)
            feat.SetField("Hemisphere", ns)
            layer.CreateFeature(feat)
            x = round(x + size, roundnum)
            x_count += 1
        y = round(y + size, roundnum)
        y_count += 1
    layer.CommitTransaction()
    return location


if __name__ == "__main__":
    global_tileset = global_region_tileset(1, "1.2")
    print('Done')