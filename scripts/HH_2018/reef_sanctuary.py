# Reef Raw Data Extraction
# Reef data were defined as OBSTRN point and polygon features with CATOBS = 5 (fish haven) and
# SEAARE point and polygon features with CATSEA = 9 (reef)
# Sanctuary bounds were extracted from the Office of National Marine Sanctuaries feature service: http://services2.arcgis.com/C8EMgrsFcRFL6LrL/arcgis/rest/services/National_Marine_Sanctuaries/FeatureServer

# Import modules
import os
import time
import arcpy
import arcgisscripting
from arcpy import env
from arcpy.sa import *
import numpy
from HSTB.ArcExt.NHSP import gridclass, Functions
from HSTB.ArcExt.NHSP import globals
from HSTB.ArcExt.NHSP.globals import Parameters, timer
# wrapping this import as eclipse parser uses python 2
exec("from HSTB.ArcExt.NHSP.globals import print")


features_1 = ['.obstrn', '.seaare']
geoms_1 = ['_point', '_polygon']
sanc_feature_service = r"http://services2.arcgis.com/C8EMgrsFcRFL6LrL/arcgis/rest/services/National_Marine_Sanctuaries/FeatureServer/0"


def extract_raw(params):
    # RAW DATA
    # Extract Raw Data: OBSTRN, PILPNT, and WRECKS Features
    return Functions.extract_enc_data_p(params, features_1, geoms_1)


def download_sanctuaries(sanc_params):
    """Downloads the polygon layer from the marine services website and returns the path
    to where it's stored which will be sanc_params.raw_filename()
    """
    fs = arcpy.FeatureSet()
    fs.load(sanc_feature_service)
    fname = sanc_params.raw_filename()
    arcpy.CopyFeatures_management(fs, fname)
    return fname


# Input Variables

query = "(catobs = '5                        ' OR catsea = '9                             ')"  # @todo There is likely a better way to program this, but the select analysis tool will not work unless all of the noted spaces are included
tank_buff_m = "4360;9260;18520;37040"
nontank_buff_m = "1852;3704"

# Definitions
# Raster Reclassification Values
# Tanker
t_rc = tank_buff_m.split(";")
tank_rc_val = "%s %s %s;%s %s %s;%s %s %s;%s %s %s" % (int(t_rc[0]), int(t_rc[0]), 5, int(t_rc[1]), int(t_rc[1]), 4, int(t_rc[2]), int(t_rc[2]), 3, int(t_rc[3]), int(t_rc[3]), 2)

# Non Tanker
nt_rc = nontank_buff_m.split(";")
ntank_rc_val = "%s %s %s; %s %s %s" % (int(nt_rc[0]), int(nt_rc[0]), 2, int(nt_rc[1]), int(nt_rc[1]), 1)


def get_raw_data(reefs, sanc):
    # RAW DATA
    # Extract OBSTRN and SEARE Data
    # Make list of raw files
    fns_raw = extract_raw(reefs)
    sanc_raw = download_sanctuaries(sanc)
    return fns_raw, sanc_raw


def project_raw_data(reefs, sanc, fns_raw, sanc_raw):
    # Project Data
    fns_prj = []
    for fn_raw in fns_raw:
        fn_prj = fn_raw + "_prj"
        fns_prj.append(Functions.check_spatial_reference(fn_raw, fn_prj, reefs.projection_name, reefs.projection_number))
    sanc_fn = Functions.check_spatial_reference(sanc_raw, sanc.vector_processed_filename(), sanc.projection_name, sanc.projection_number)
    return fns_prj, sanc_fn


def create_vector_processed(reefs, fns_prj):
    # VECTOR PROCESSED
    # Select OBSTRN with CATOBS = 5 and SEAARE with CATSEA = 9
    fns_reef_vp = []
    for geom, fn_prj in zip(geoms_1, fns_prj):
        fn_vp = reefs.vector_processed_filename(geom)
        arcpy.Select_analysis(fn_prj, fn_vp, query)
        fns_reef_vp.append(fn_vp)
    return fns_reef_vp


def create_raster(reefs, fns_reef_vp, sanc, sanc_fn, buff_m, name):
    # Reefs
    reef_r = reefs.raster_filename(name)
    reef_buff_v = []
    reef_buff_r = []
    for i, fn_vp in enumerate(fns_reef_vp):
        fn_buff = reefs.working_filename("temp_Buff" + "%d" % (i))
        fn_ras_t = reefs.working_rastername("_r" + "%s" % (i))
        arcpy.MultipleRingBuffer_analysis(fn_vp, fn_buff, buff_m, "Meters", "value", "ALL", "FULL")
        arcpy.PolygonToRaster_conversion(fn_buff, "value", fn_ras_t, "MAXIMUM_AREA", "value", reefs.cell_size)
        reef_buff_v.append(fn_buff)
        reef_buff_r.append(fn_ras_t)
    # Sanctuaries
    sanc_buff_v = sanc.working_filename("temp_Buff_sanc")
    sanc_buff_r = sanc.working_rastername("_rs")
    arcpy.MultipleRingBuffer_analysis(sanc_fn, sanc_buff_v, buff_m, "Meters", "value", "ALL", "FULL")
    arcpy.PolygonToRaster_conversion(sanc_buff_v, "value", sanc_buff_r, "MAXIMUM_AREA", "value", sanc.cell_size)

    # Mosaic to New Raster - Reef and Sanctuary Buffered Rasters
    # Join Tanker Reef and Sanctuary Files (Multi-Buffer and Raster)
    arcpy.MosaicToNewRaster_management(reef_buff_r + [sanc_buff_r], os.path.split(reef_r)[0], os.path.split(reef_r)[1], reefs.projection_number, "32_BIT_SIGNED", str(reefs.cell_size), "1", "MINIMUM", "FIRST")
    for flist in (reef_buff_v, reef_buff_r, [sanc_buff_v, sanc_buff_r]):
        for del_file in flist:
            arcpy.Delete_management(del_file)
    return reef_r


# def create_rasters(reefs, fns_reef_vp, sanc, sanc_fn):
#     # RASTER
#     # Tankers
#     # Reefs
#     reef_tank_r = reefs.raster_filename("tk")
#     reef_buff_v = []
#     reef_buff_r = []
#     for i, fn_vp in enumerate(fns_reef_vp):
#         fn_buff = reefs.working_filename("temp_Buff" + "%d" % (i))
#         fn_ras_t = reefs.working_rastername("_r" + "%s" % (i))
#         arcpy.MultipleRingBuffer_analysis(fn_vp, fn_buff, tank_buff_m, "Meters", "value", "ALL", "FULL")
#         arcpy.PolygonToRaster_conversion(fn_buff, "value", fn_ras_t, "MAXIMUM_AREA", "value", 500)
#         reef_buff_v.append(fn_buff)
#         reef_buff_r.append(fn_ras_t)
#     # Sanctuaries
#     sanc_buff_v = sanc.working_filename("temp_Buff_sanc")
#     sanc_buff_r = sanc.working_rastername("_r3")
#     arcpy.MultipleRingBuffer_analysis(sanc_fn, sanc_buff_v, tank_buff_m, "Meters", "value", "ALL", "FULL")
#     arcpy.PolygonToRaster_conversion(sanc_buff_v, "value", sanc_buff_r, "MAXIMUM_AREA", "value", 500)
#
#     # Mosaic to New Raster - Reef and Sanctuary Buffered Rasters
#     # Join Tanker Reef and Sanctuary Files (Multi-Buffer and Raster)
#     arcpy.MosaicToNewRaster_management(reef_buff_r + [sanc_buff_r], os.path.split(reef_tank_r)[0], os.path.split(reef_tank_r)[1], reefs.projection_number, "32_BIT_SIGNED", str(reefs.cell_size), "1", "MINIMUM", "FIRST")
#     for flist in (reef_buff_v, reef_buff_r, [sanc_buff_v, sanc_buff_r]):
#         for del_file in flist:
#             arcpy.Delete_management(del_file)
#
#     # Non Tankers
#     reef_notank_r = reefs.raster_filename("ntk")
#     reef_buff_nt_v = []
#     reef_buff_nt_r = []
#     for i, fn_vp in enumerate(fns_reef_vp):
#         fn_buff_nt = reefs.working_filename("buff_nt" + "%d" % (i))
#         fn_ras_nt_t = reefs.working_rastername("_nt" + "%s" % (i))
#         arcpy.MultipleRingBuffer_analysis(fn_vp, fn_buff_nt, nontank_buff_m, "Meters", "value", "ALL", "FULL")
#         arcpy.PolygonToRaster_conversion(fn_buff_nt, "value", fn_ras_nt_t, "MAXIMUM_AREA", "value", 500)
#         reef_buff_nt_v.append(fn_buff_nt)
#         reef_buff_nt_r.append(fn_ras_nt_t)
#
#     # Sanctuaries
#     sanc_buff_nt_v = sanc.working_filename("Buff_nt")
#     sanc_buff_nt_r = sanc.working_rastername("t_nt")
#     arcpy.MultipleRingBuffer_analysis(sanc_fn, sanc_buff_nt_v, nontank_buff_m, "Meters", "value", "ALL", "FULL")
#     arcpy.PolygonToRaster_conversion(sanc_buff_nt_v, "value", sanc_buff_nt_r, "MAXIMUM_AREA", "value", 500)
#
#     # Mosaic to New Raster - Reef and Sanctuary Buffered Rasters
#     # Join non-Tanker Reef and Sanctuary Files (Multi-Buffer and Raster)
#     arcpy.MosaicToNewRaster_management(reef_buff_nt_r + [sanc_buff_nt_r], os.path.split(reef_notank_r)[0], os.path.split(reef_notank_r)[1], reefs.projection_number, "32_BIT_SIGNED", str(reefs.cell_size), "1", "MINIMUM", "FIRST")
#     for flist in (reef_buff_nt_v, reef_buff_nt_r, [sanc_buff_nt_v, sanc_buff_nt_r]):
#         for del_file in flist:
#             arcpy.Delete_management(del_file)
#
#     return reef_tank_r, reef_notank_r


def classify_raster(reefs, rastername, classified_name, eez):
    reef_tank_rc = reefs.raster_classified_filename(classified_name)
    tank_rc_temp = arcpy.sa.Reclassify(rastername, "VALUE", tank_rc_val, "DATA")
    arcpy.MosaicToNewRaster_management([tank_rc_temp, eez], os.path.split(reef_tank_rc)[0], os.path.split(reef_tank_rc)[1], reefs.projection_number, "32_BIT_SIGNED", str(reefs.cell_size), "1", "MAXIMUM", "FIRST")
    return reef_tank_rc

# def classify_rasters(input_rasters, reefs):
#     # RASTER CLASSIFIED
#     # Tanker
#     reef_tank_rc = reefs.raster_classified_filename("tk")
#     tank_rc_temp = arcpy.sa.Reclassify(reef_tank_r, "VALUE", tank_rc_val, "DATA")
#     arcpy.MosaicToNewRaster_management([tank_rc_temp, eez_1], os.path.split(reef_tank_rc)[0], os.path.split(reef_tank_rc)[1], reefs.projection_number, "32_BIT_SIGNED", str(reefs.cell_size), "1", "MAXIMUM", "FIRST")
#
#     # Non Tanker
#     reef_notank_rc = reefs.raster_classified_filename("ntk")
#     ntank_rc_temp = arcpy.sa.Reclassify(reef_notank_r, "VALUE", ntank_rc_val, "DATA")
#     arcpy.MosaicToNewRaster_management([ntank_rc_temp, eez_0], os.path.split(reef_notank_rc)[0], os.path.split(reef_notank_rc)[1], reefs.projection_number, "32_BIT_SIGNED", str(reefs.cell_size), "1", "MAXIMUM", "FIRST")


def execute(ini):
    reefs = Parameters("Reefs", ini)
    sanc = Parameters("Sanctuary", ini)
    ais = Parameters("AIS", ini)
    eez_0 = ini["AUX"]["eez_0_ras"]
    eez_1 = ini["AUX"]["eez_1_ras"]

    fns_raw = [reefs.raw_filename('_point'), reefs.raw_filename('_polygon')]
    for fn in fns_raw:
        if not arcpy.Exists(fn):
            print("Extracting from the ENC source")
            fns_raw = extract_raw(reefs)
            break
    sanc_raw = sanc.raw_filename()
    if not arcpy.Exists(sanc_raw):
        print("Downloading from sanctuary feature service")
        sanc_raw = download_sanctuaries(sanc)

    fns_prj, sanc_fn = project_raw_data(reefs, sanc, fns_raw, sanc_raw)
    fns_reef_vp = create_vector_processed(reefs, fns_prj)
    # reef_tank_r, reef_notank_r = create_rasters(reefs, fns_reef_vp, sanc, sanc_fn)
    print("Buffering and creating tanker rasters")
    reef_tank_r = create_raster(reefs, fns_reef_vp, sanc, sanc_fn, tank_buff_m, "tk")
    print("Buffering and creating non-tanker rasters")
    reef_notank_r = create_raster(reefs, fns_reef_vp, sanc, sanc_fn, nontank_buff_m, "ntk")

    # classify_rasters(input_rasters, reefs)
    print("Classifying tanker rasters")
    reef_tank_rc = classify_raster(reefs, reef_tank_r, "tk", eez_1)
    print("Classifying non-tanker rasters")
    reef_notank_rc = classify_raster(reefs, reef_notank_r, "ntk", eez_0)

    reef_tank_rf = reefs.raster_final_filename("tk")
    reef_notank_rf = reefs.raster_final_filename("ntk")
    reef_tank_rf_t = reefs.working_rastername("tk_rft")

    ais_tanker = ais.raster_classified_filename("tank_1")

    # Raster Final
    # Tanker
    print("Finalizing tanker rasters")
    tank_rf_r = arcpy.sa.ExtractByMask(reef_tank_rc, eez_0)
    tank_rf_r.save(reef_tank_rf_t)

    # Clip Tanker to Extents of Tanker Traffic, such that, if no tanker (assume all tanker traffic captured by AIS, then consequence to tanker is 0)
    ais_tanker_binary_ras = arcpy.sa.Raster(ais_tanker)
    reef_tank_rf_t_ras = arcpy.sa.Raster(reef_tank_rf_t)

    ais_reefs_tank = ais_tanker_binary_ras * reef_tank_rf_t_ras
    ais_reefs_tank.save(reef_tank_rf)

    # Non Tanker
    print("Finalizing non-tanker rasters")
    tank_rf_r = arcpy.sa.ExtractByMask(reef_notank_rc, eez_0)
    tank_rf_r.save(reef_notank_rf)

    # Delete Intermediate Files
    del_files = []
    for del_file in del_files:
        arcpy.Delete_management(del_file)


def main():
    execute(globals.initial_values)
