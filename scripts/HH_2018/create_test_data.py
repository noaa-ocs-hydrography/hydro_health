import os
import importlib
import math
import shutil

import arcpy
import arcgisscripting

from HSTB.ArcExt.NHSP import globals
from HSTB.ArcExt.NHSP import gridclass
from HSTB.ArcExt.NHSP import groundings
from HSTB.ArcExt.NHSP import reported_error
from HSTB.ArcExt.NHSP import human_debris
from HSTB.ArcExt.NHSP import known_hazards
from HSTB.ArcExt.NHSP import reef_sanctuary
from HSTB.ArcExt.NHSP import sar
from HSTB.ArcExt.NHSP import storms
from HSTB.ArcExt.NHSP import eez
from HSTB.ArcExt.NHSP import Create_Grid

# load the master/original data paths
importlib.reload(globals)
importlib.reload(gridclass)
importlib.reload(reported_error)
importlib.reload(groundings)
importlib.reload(human_debris)
importlib.reload(known_hazards)
importlib.reload(reef_sanctuary)
importlib.reload(storms)
importlib.reload(sar)
exec("from HSTB.ArcExt.NHSP.globals import print")

wgs_sr = arcpy.SpatialReference(4326)  # wgs 1984


def clip(fname, clip_area):
    """ Arc doesn't let you clip to the same layer name so this convenience function:
    renames the original data,
    clips the data back to the original name,
    deletes the renamed data
    """
    # print("inside of clip function", fname)
    d = arcpy.Describe(fname)
    is_raster = "raster" in d.dataType.lower()

    if "c:" != fname[:2].lower():
        raise Exception("WHOA!!!  Only local temp data can be made -- you might overwrite network data accidentally")
    tmp_name = fname + "_temp_del"
    # print("renaming", fname, tmp_name)
    try:
        arcpy.Rename_management(fname, tmp_name)
    except arcgisscripting.ExecuteError as e:
        if is_raster:
            # if it's a raster then there is a filename limit of 13 characters, so let's try and shorten the name
            dirname, basename = os.path.split(fname)
            tmp_name = os.path.join(dirname, (basename + "_temp_del")[:13])
            if os.path.basename(tmp_name) == basename:
                # name was exactly 13 characters, so change the end characters
                tmp_name = tmp_name[:-4] + "_del"
            # now if it fails it should be write permissions or something else
            arcpy.Rename_management(fname, tmp_name)
        else:
            raise e
    print(("clipping", tmp_name, is_raster, fname))
    if is_raster:
        # have to reproject the clip area into the right coordinate system and then make a string of xmin, ymin, xmax, ymax
        c = clip_area.projectAs(d.spatialReference)
        clip_rect = " ".join([str(c) for c in (c.extent.XMin, c.extent.YMin, c.extent.XMax, c.extent.YMax)])
        arcpy.Clip_management(tmp_name, clip_rect, fname)
    else:
        arcpy.Clip_analysis(tmp_name, clip_area, fname)
    arcpy.Delete_management(tmp_name)


def copy_given_data(src_ini, dest_ini, clipping_polygon):
    # Create general datasets
    # copy the eez vector (supplied by Lucy Hick originally) and make a raster of the vector at the resolution of the test dataset
    aux = globals.Parameters("AUX", src_ini)
    aux_test = globals.Parameters("AUX", dest_ini)

    arcpy.Copy_management(aux["eez_buffered"], aux_test["eez_buffered"])
    clip(aux_test["eez_buffered"], clipping_polygon)
    arcpy.PolygonToRaster_conversion(aux_test["eez_buffered"], "Dissolve_ID", aux_test["eez_buffered_ras"], "CELL_CENTER", cellsize=aux_test.cell_size)
    eez.create(aux_test)

    # Get the CATZOC, survey quality layers
    # currently supplied by SurveyQuality scripts but hopefully delivered from National Bathymetric Source Project or MCD authoritative source in the future.
    arcpy.Copy_management(aux["catzoc_bcdu_raster"], aux_test["catzoc_bcdu_raster"])
    clip(aux_test["catzoc_bcdu_raster"], clipping_polygon)


def create_directories(dest_ini):
    base_test = globals.BaseParameters(dest_ini)
    ac_test = globals.Parameters("ActiveCaptain", dest_ini)
    ground_test = globals.Parameters("Groundings", dest_ini)
    ports_test = globals.Parameters("Ports", dest_ini)
    reperr_test = globals.Parameters("RepErr", dest_ini)
    debris_test = globals.Parameters("HumanDebris", dest_ini)
    haz_test = globals.Parameters("KnownHazards", dest_ini)
    reefs_test = globals.Parameters("Reefs", dest_ini)
    sanc_test = globals.Parameters("Sanctuary", dest_ini)
    storms_test = globals.Parameters("Storms", dest_ini)

    # make all the generic data directories
    for pth in (base_test.raw_dir, base_test.vector_processed_dir, base_test.vector_dir, base_test.vector_final_dir,
                base_test.raster_dir, base_test.raster_classified_dir, base_test.raster_final_dir, base_test.aux_dir, base_test.grid_dir):
        os.makedirs(pth, exist_ok=True)
    for gdb in (base_test.raw_gdb, base_test.vector_processed_gdb, base_test.vector_gdb, base_test.vector_final_gdb,
                base_test.raster_gdb, base_test.raster_classified_gdb, base_test.raster_final_gdb, base_test.aux_gdb, base_test.grid_gdb):
        try:
            arcpy.CreateFileGDB_management(*os.path.split(gdb))
        except arcgisscripting.ExecuteError:
            pass
    for dtype in (ac_test, ground_test, ports_test, reperr_test, debris_test, haz_test, reefs_test, sanc_test, storms_test):
        try:
            os.makedirs(os.path.dirname(dtype.working_gdb), exist_ok=True)
            arcpy.CreateFileGDB_management(*os.path.split(dtype.working_gdb))
        except arcgisscripting.ExecuteError:
            pass


def copy_grid_outlines(src_ini, dest_ini, clipping_polygon):
    # copy the vector outlines of the eez based areas which are being used as sub-areas for computational speed/memory reasons
    # arcpy.Copy_management(src["grid_ws"], dest["grid_ws"])
    src = globals.Parameters("Input", src_ini)
    dest = globals.Parameters("Input", dest_ini)
    for i in range(1, 14):
        md = "Model_Extents_%02d" % i
        test_md = os.path.join(dest["grid_ws"], md)
        arcpy.Copy_management(os.path.join(src["grid_ws"], md), test_md)
        clip(test_md, clipping_polygon)
    arcpy.Copy_management(os.path.join(src["grid_ws"], src["US_shoreline"]),
                          os.path.join(dest["grid_ws"], dest["US_shoreline"]))
    Create_Grid.create_grids(dest)
    # clip(os.path.join(dest["grid_ws"], ), clip_area)


def copy_ais(src_ini, dest_ini, clipping_polygon):
    # The ais data was download from axiom at http://ais.axds.co/
    # They were then combined and clossified
    # since we aren't sure this is the future way to get data (let alone calculate UKC)
    # I am copying the data as is.  See the Axiom_AIS script for how they are made.
    ais = globals.Parameters("AIS", src_ini)
    ais_test = globals.Parameters("AIS", dest_ini)

    for fname in ("a_ac_u", "tank_u", "pass_u", "ntk_ac", "np_ac", "tank_1", "pass_0", "sar_p", "np_ac0", "sar_np"):
        arcpy.Clip_analysis(ais.raster_classified_filename(fname), clipping_polygon, ais_test.raster_classified_filename(fname))

    for fname in ("a_ac_u", "tanku", "passu", "ntk_ac", "np_ac", "sar_p", "sar_np"):
        arcpy.Clip_analysis(ais.raster_final_filename(fname), clipping_polygon, ais_test.raster_final_filename(fname))


def copy_active_captain(src_ini, dest_ini, clipping_polygon):
    # Active Captain
    ac = globals.Parameters("ActiveCaptain", src_ini)
    ac_test = globals.Parameters("ActiveCaptain", dest_ini)
    arcpy.Clip_analysis(ac.raw_filename(), clipping_polygon, ac_test.raw_filename())


def copy_groundings(src_ini, dest_ini, clipping_polygon):
    # Groundings
    ground = globals.Parameters("Groundings", src_ini)
    ground_test = globals.Parameters("Groundings", dest_ini)
    ground_raw = os.path.join(ground.raw_dir, ground["csv_data"])  # User-Defined
    x_coords = ground["x_coords"]  # User-Defined
    y_coords = ground["y_coords"]  # User-Defined
    gvp = groundings.create_vp(ground_raw, x_coords, y_coords, ground_test)
    if gvp:
        clip(gvp, clipping_polygon)
    else:
        print("Need to have an open map for MakeXYEventLayer inside Groundings to work")


def copy_sar(src_ini, dest_ini, clipping_polygon):
    sar_src = globals.Parameters("SAR", src_ini)
    sar_test = globals.Parameters("SAR", dest_ini)
    shutil.copyfile(os.path.join(sar_src.raw_dir, sar_src["csv_data"]), os.path.join(sar_test.raw_dir, sar_test["csv_data"]))
    sar_v = sar.create_vector(sar_test)
    clip(sar_v, clipping_polygon)


def copy_ports(src_ini, dest_ini, clipping_polygon):
    # Ports
    ports = globals.Parameters("Ports", src_ini)
    ports_test = globals.Parameters("Ports", dest_ini)
    arcpy.Clip_analysis(ports.raw_filename("Bounds"), clipping_polygon, ports_test.raw_filename("Bounds"))
    arcpy.Clip_analysis(ports.raw_filename("USACE"), clipping_polygon, ports_test.raw_filename("USACE"))


def copy_reported_error(src_ini, dest_ini, clipping_polygon):
    # Reported Error
    reperr = globals.Parameters("RepErr", src_ini)
    reperr_test = globals.Parameters("RepErr", dest_ini)
    fnames = reported_error.extract_raw(reperr_test)
    for fname in fnames:
        clip(fname, clipping_polygon)


def copy_human_debris(src_ini, dest_ini, clipping_polygon):
    # Human Debris
    debris = globals.Parameters("HumanDebris", src_ini)
    debris_test = globals.Parameters("HumanDebris", dest_ini)
    fnames = human_debris.extract_raw(debris_test)
    for fname in fnames:
        clip(fname, clipping_polygon)


def copy_known_hazards(src_ini, dest_ini, clipping_polygon):
    # Known Hazards
    haz = globals.Parameters("KnownHazards", src_ini)
    haz_test = globals.Parameters("KnownHazards", dest_ini)
    fnames = known_hazards.extract_raw(haz_test)
    for fname in fnames:
        clip(fname, clipping_polygon)


def copy_reefs_sanctuaries(src_ini, dest_ini, clipping_polygon):
    # Reefs and Sanctuaries
    reefs = globals.Parameters("Reefs", src_ini)
    reefs_test = globals.Parameters("Reefs", dest_ini)
    fnames = []
    exist = [arcpy.Exists(reefs.raw_filename(geom)) for geom in reef_sanctuary.geoms_1]
    if all(exist):
        for geom in reef_sanctuary.geoms_1:
            fn = reefs.raw_filename(geom)
            fnames.append(fn)
            arcpy.Copy_management(fn, reefs_test.raw_filename(geom))
    else:
        fnames = reef_sanctuary.extract_raw(reefs_test)

    sanc = globals.Parameters("Sanctuary", src_ini)
    sanc_test = globals.Parameters("Sanctuary", dest_ini)
    if arcpy.Exists(sanc.raw_filename()):
        arcpy.Copy_management(sanc.raw_filename(), sanc_test.raw_filename())
        sancname = sanc_test.raw_filename()
    else:
        sancname = reef_sanctuary.download_sanctuaries(sanc_test)
    fnames.append(sancname)
    for fname in fnames:
        clip(fname, clipping_polygon)


def copy_storms(src_ini, dest_ini, clipping_polygon):
    # Storms
    with globals.timer("storms") as t:
        storms_params = globals.Parameters("Storms", src_ini)
        storms_test = globals.Parameters("Storms", dest_ini)
        merged_name = storms_test.working_filename("c_windswath_merged")
        i_dataset2_fldr = os.path.join(storms_params.raw_dir, storms_params["swaths"])  # folder containing shape files of polygons of storms
        i_dataset1_file = os.path.join(storms_params.raw_dir, storms_params["tracks"])  # shapefile containing polyline tracks of storms
        t.msg("merging storm swaths - with projection")
        storms.merge_storm_swaths(i_dataset2_fldr, merged_name)
        t.msg("clipping storm swaths")
        clip(merged_name, clipping_polygon)
        c_line_dissolved = storms_test.working_filename(storms.str_c_line_dissolved)
        t.msg("dissolve on storm lines")
        arcpy.Dissolve_management(i_dataset1_file, c_line_dissolved, storms.list_dissolved, "", "MULTI_PART", "DISSOLVE_LINES")
        t.msg("clip storm lines")
        clip(c_line_dissolved, clipping_polygon)


def create_data(ini_filename, test_dir, clipping_polygon, cell_size=0):
    # create a set of paths where the clipped test data will be stored
    # if 1:
    src_ini = globals.IniFile(ini_filename)
    dest_ini = globals.IniFile(ini_filename)
    dest_ini["Drives"]["HSD_DATA"] = test_dir
    if cell_size:
        dest_ini["Input"]["cell_size"] = str(cell_size)

    # dest_ini.save_as(os.path.join(os.path.dirname(globals.__file__), "test_NHSP.ini"))
    if os.path.normpath(os.path.normcase(src_ini["Drives"]["HSD_DATA"])) == os.path.normpath(os.path.normcase(dest_ini["Drives"]["HSD_DATA"])):
        raise Exception("The directory to create test data can not be the same as contained in the original ini file")
    with globals.timer("Finished copy/clipping of data", "Starting copy/clipping of data") as t:
        create_directories(dest_ini)
        t.msg("Made directories and empty GDBs")
        copy_given_data(src_ini, dest_ini, clipping_polygon)
        t.msg("Copied input layers")
        copy_grid_outlines(src_ini, dest_ini, clipping_polygon)
        t.msg("copied grid outlines")
        copy_active_captain(src_ini, dest_ini, clipping_polygon)
        t.msg("copied active captain")
        copy_groundings(src_ini, dest_ini, clipping_polygon)
        t.msg("copied groundings")
        copy_ports(src_ini, dest_ini, clipping_polygon)
        t.msg("copied ports")
        copy_reported_error(src_ini, dest_ini, clipping_polygon)
        t.msg("copied reported error data")
        copy_human_debris(src_ini, dest_ini, clipping_polygon)
        t.msg("copied human debris")
        copy_known_hazards(src_ini, dest_ini, clipping_polygon)
        t.msg("copied hazards data")
        copy_reefs_sanctuaries(src_ini, dest_ini, clipping_polygon)
        t.msg("copied reefs")
        copy_sar(src_ini, dest_ini, clipping_polygon)
        t.msg("copied Search and Rescue")
        raise Exception("need to debug the ais copy")
        copy_ais(src_ini, dest_ini, clipping_polygon)
        t.msg("copied AIS data")

        raise Exception("need to debug the storms copy, it isn't getting a projection it seems for data so clip isn't working")
        copy_storms(src_ini, dest_ini, clipping_polygon)
        t.msg("copied storms")
    # Currents

    # Search and Rescue


def main():
    test_dir = "c:\\data\\ArcGIS\\NHSP_TestData\\"
    cell_size = 500
    test_clip = True

    if test_clip:
        north = 30.0
        south = 28.5
        east = -94.0
        west = -95.5
    else:
        north = 90.0
        south = -90.0
        east = 180.0
        west = -180.0

    clip_area = arcpy.Polygon(arcpy.Array([arcpy.Point(east, south, 0),
                                           arcpy.Point(east, north, 0),
                                           arcpy.Point(west, north, 0),
                                           arcpy.Point(west, south, 0)]), wgs_sr)
    create_data(os.path.join(os.path.dirname(globals.__file__), "NHSP.ini"), test_dir, clip_area)
