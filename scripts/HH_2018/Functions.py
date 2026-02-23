import os
import arcpy
import arcgisscripting
import collections
import traceback

# Superseded CATZOC and Year Output
# @todo: Ask Barry to add in check for raster filename length (e.g. can not exceed 13 char). Then, delete output filenames [0:4]


def query_csv(csv_filepath, x, y, params, query="", prj=4326):
    """ Create a vector processed gdb layer from the supplied csv data using an SQL query and returns the path to the created data
    raw_data is a path to csv data
    x, y are the field names to use for longitude and latitude
    params is a globals.Parameters instance which generates the filenames
    query is an SQL string used to convert the raw data to vector processed layer
    prj is wgs84 by default
    """
    try:
        raw_lyr = os.path.basename(params.raw_filename())
        arcpy.MakeXYEventLayer_management(csv_filepath, x, y, raw_lyr, prj)
        v = params.vector_filename()
        arcpy.CopyFeatures_management(raw_lyr, v)
        if query:
            vp = params.vector_processed_filename()
            arcpy.Select_analysis(v, vp, query)
            return vp
        else:
            return v
    except arcgisscripting.ExecuteError:
        traceback.print_exc()
        print("Unable to query input file, verify New Map is opened in ArcPro. e.g. Insert --> New Map.")
        return ""


def generate_polygon_bounds(input_filepath, output_filepath):

    # Generate Bounds of Final Layer
    dissolve_field = "dis_field"
    arcpy.AddField_management(input_filepath, dissolve_field, "double")
    arcpy.Dissolve_management(input_filepath, output_filepath, dissolve_field)


def calc_superseded_catzoc_year(input_filename, catzoc_fieldname, year_fieldname, output_gdb, output_folder, raster_cell_size, priority_fieldname, priority, num_priorities):
    '''Outputs Final CATZOC raster and Year raster. Input raster must contain calculated priority field

    input_filename = path to input raster with priority field (raw string)
    catzoc_fieldname = field name of final catzoc field (string)
    year_fieldname = field name of final year field (string)
    output_folder = path to final output folder where rasters will be saved (raw string)
    output_gbb = path to final geodatabase where feature classes will be saved (should create temporary geodatabase) (raw string)
    raster_cell_size = cell size of output raster in meters (string)
    priority = priority of input_filename (e.g. assuming dataset 1 is of higher quality and should supersede datsaet 2, then priority for dataset 1 would be 1 and for datsaet 2, priority would be 2).
    num_priorities = number of input datasets to be prioritized
    '''
    # Variables
    priority_mulitplier = 10**(4 + (num_priorities - priority))

    # Check CATZOC and Year fields to verify they are integers, if not, add new integer field
    catzoc_year_final_fieldnames = {"catzoc_field_final": catzoc_fieldname, "year_field_final": year_fieldname}
    fields = arcpy.ListFields(input_filename)
    for field in fields:
        for catzoc_yr_handle, catzoc_year_field in list(catzoc_year_final_fieldnames.items()):
            if str(field.name) == catzoc_year_field and field.type != 'Double':
                new_catzoc_year_fieldname = catzoc_year_field + "Final"
                arcpy.AddField_management(input_filename, new_catzoc_year_fieldname, "double")
                arcpy.CalculateField_management(input_filename, new_catzoc_year_fieldname, "!" + catzoc_year_field + "!")
                catzoc_year_final_fieldnames[catzoc_yr_handle] = new_catzoc_year_fieldname

    # Add unique priority field
    priority_calc = "(%d * (6-!%s!)) + !%s!" % (priority_mulitplier, catzoc_year_final_fieldnames["catzoc_field_final"], catzoc_year_final_fieldnames["year_field_final"])
    arcpy.AddField_management(input_filename, priority_fieldname, "double")
    arcpy.CalculateField_management(input_filename, priority_fieldname, priority_calc, "PYTHON3")


# Check spatial reference of input polygon, re-project if not desired coordinate system
def check_spatial_reference(input_filepath, output_filepath, des_prj_name, des_prj_code):
    '''Check Spatial Reference will identify if the input layer is in the desired projection. If not, the input file will be re-projected to the desired spatial reference

    input_filepath = string of the geodatabase where the input layer is stored
    output_filepath = string of the geodatabase where the output layer will be saved if needed
    des_prj_name = string of name of desired spatial reference according to http://spatialreference.org/ref/esri/
    des_prj_code = integer of keycode for desired spatial reference according to http://spatialreference.org/ref/esri/

    The re-projected file will be exported to the user-identified geodatabase with the des_prj_name appended to the filename
    '''

    desc = arcpy.Describe(input_filepath)
    sr = desc.spatialReference

    if sr.name != des_prj_name:
        print(("Re-projecting " '%s' % (input_filepath)))
        arcpy.Project_management(input_filepath, output_filepath, des_prj_code)
    else:
        print((input_filepath + " is in the desired coordinate system: " + des_prj_name))  # + " Copying feature with _"+des_prj_name)
        output_filepath = input_filepath
    return output_filepath


def Make_Selection(gdb, input_filename, query, output_filename):
    '''Make Selection will query an input layer within a user-defined geodatabase and then export the queried layer to the same geodatabase
    gdb = string of the geodatabase where the input layer is stored as well as where the queried layer will be exported
    input_filename = string of the name of the input layer
    query = query command, e.g. "gs <> 'Fishing Vessel'"
    output_filename = string of the name of the exported layer
    '''

    try:
        input = os.path.join(gdb, input_filename)
        output = os.path.join(gdb, output_filename)
        input_temp = "temp"

        arcpy.MakeFeatureLayer_management(input, input_temp)
        arcpy.SelectLayerByAttribute_management(input_temp, "NEW_SELECTION", query)
        arcpy.CopyFeatures_management(input_temp, output)
        arcpy.SelectLayerByAttribute_management(input_temp, "CLEAR_SELECTION")
        return output
    except arcgisscripting.ExecuteError:
        print("Unable to query input file, verify New Map is opened in ArcPro. e.g. Insert --> New Map.")

# Make_Selection("H:\Christy\Groundings\Groundings.gdb", "Groundings_VECTOR", "gs <> 'Fishing Vessel' AND gs <> 'Recreational'  AND status = 'Grounding'", "Groundings_VECTOR_PROCESSED2")


def extract_enc_data(enc_sde, ub_sde, output_dir, features, geoms, base_filename, prj, cleanup=True):
    """Maintains backward compatibilty for existing code.  everything will end up in output directory and the final output file will start with base_filename
    """
    def filenamer(fname=""):
        return os.path.join(output_dir, fname)

    def output_filenamer(fname=""):
        return os.path.join(output_dir, base_filename + fname)
    return extract_enc_data_fn(enc_sde, ub_sde, filenamer, output_filenamer, features, geoms, prj, cleanup)


def extract_enc_data_p(params, features, geoms, cleanup=True):
    """Calls extract_enc_data_fn with the 'working' gdb filename creator for temporary files and RAW for output files
    """
    enc_sde = os.path.join(params.sde_dir, params.ini["DBConn"]["enc"])
    ub_sde = os.path.join(params.sde_dir, params.ini["DBConn"]["scale_usage_bands"])
    if not os.path.exists(enc_sde):
        raise Exception("Path to enc sde file not found:" + enc_sde)
    if not os.path.exists(enc_sde):
        raise Exception("Path to scale usage bands sde file not found:" + ub_sde)
    return extract_enc_data_fn(enc_sde, ub_sde, params.working_filename, params.raw_filename, features, geoms, params.projection_number, cleanup)


def extract_enc_data_fn(enc_sde, ub_sde, temp_filenames, output_filenames, features, geoms, prj, cleanup=True):
    r'''This function extracts user-specified features from the ENC Viewer database, merges all features of identical geometry, clips features to best scale ENC usage band, and then
    exports a merged feature layer for each geometry type.

    NOTE: This script assumes the data naming of the enc viewer database structure is as follows: ocsdev.ENCBAND.s57Attribute_geometry. e.g. ocsdev.approach.obstrn_point
          This script assumes the best scale usage bands are named as follows: ENC_BestCoverage_UB1. If this is not the case adjust clip_out pathname.

    enc_sde = string of file location for ENC Viewer Database Connection with read command.
    ub_sde = string of file location for Best Scale Usage Band database connection with read command.  r"C:\Users\Christina.Fandel\AppData\Roaming\ESRI\Desktop10.4\ArcCatalog\postgresql_dev_encd_viewer.sde"
    filename_generator = function that returns a full path to where data should be stored (similar to the xxx_filename from globals.Parameters)
    features = list of strings of desired features as defined in ENC Viewer database. e.g. ['.obstrn', '.pilpnt', '.uwtroc', '.wrecks']
    geoms = list of strings of desired geometries as defined in ENC Viewer database. e.g. ['_point', '_line', '_polygon']
    prj = integer of projection keycode (http://pro.arcgis.com/en/pro-app/arcpy/classes/pdf/projected_coordinate_systems.pdf)
    cleanup = deletes the temporary layers downloaded from the database.  Set to False to inspect the raw data in case of debugging.
    '''

    OVERVIEW, GENERAL, COASTAL, APPROACH, HARBOR, BERTHING = ['overview', 'general', 'coastal', 'approach', 'harbor', 'berthing']
    bands = collections.OrderedDict([(OVERVIEW, '1'), (GENERAL, '2'), (COASTAL, '3'), (APPROACH, '4'), (HARBOR, '5'), (BERTHING, '6')])

    L = []
    merged_feature_layers = {}
    database_tables = {}
    for geom in geoms:
        merged_feature_layers[geom] = []
    for band, uband in list(bands.items()):
        database_tables[band] = {}
        database_tables[band] = {}
        for geom in geoms:
            database_tables[band][geom] = []
            for feat in features:
                geom_exist = enc_sde + '\\' + 'ocsdev.' + band + feat + geom
                if arcpy.Exists(geom_exist):
                    database_tables[band][geom].append(geom_exist)
            if database_tables[band][geom]:
                merge_out = temp_filenames(band + geom + '_merge')
                arcpy.Merge_management(database_tables[band][geom], merge_out)
                band_geom_mergename = temp_filenames(band + geom + '_merge_' + 'UB' + uband + 'Clip')
                clip_out = os.path.join(ub_sde, "ENC_BestCoverage_UB" + uband)
                arcpy.Clip_analysis(merge_out, clip_out, band_geom_mergename)
                merged_feature_layers[geom].append(band_geom_mergename)
                L.append(merge_out)
                L.append(band_geom_mergename)

    # Merge all bands together
    fnames = []
    for geom in geoms:
        merged_filename = temp_filenames("All_UB" + geom + '_merge')
        arcpy.Merge_management(merged_feature_layers[geom], merged_filename)
        geom_fname = output_filenames(geom)
        arcpy.Project_management(merged_filename, geom_fname, prj)
        if cleanup:
            arcpy.Delete_management(merged_filename)
        fnames.append(geom_fname)

    # Delete Extraneous Data
    for ll in L:
        if cleanup:
            arcpy.Delete_management(ll)

    return fnames

# extract_enc_data(enc_sde_1, ub_sde_1, output_gdb_1, features_1, geoms_1, output_fn_1, prj_1)
