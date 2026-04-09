# Mosaic AXIOM AIS Data to single file

# Contains AllShips and TotalVesselTraffic
# Contains AllShips and UniqueVesselTraffic

# Contains TankerShips and TotalVesselTraffic
# Contains TankerShips and UniqueVesselTraffic

# Contains CargoShips and TotalVesselTraffic
# Contains CargoShips and UniqueVesselTraffic

# Contains OtherShips and TotalVesselTraffic
# Contains OtherShips and UniqueVesselTraffic

# Contains PassengerShips and TotalVesselTraffic
# Contains PassengerShips and UniqueVesselTraffic

# Import modules
import os
import arcpy
from HSTB.ArcExt.NHSP.globals import Parameters
ais = Parameters("AIS")
aux = Parameters("AUX")
ac = Parameters("ActiveCaptain")

# User-Defined Variables
# 2014 datasets- merge 2014 passenger length dataset with 2016 version (note, 2016 version does not have passenger vessel length values. Assumed to be smallest length, 10 m).
ais_pass_len_rc_2014 = r'Y:\Hydro_Health\2016\Raster_Classified\pass_len_clas'

# Define Variables
eez_1 = aux["eez_1_ras"]
ais_raw_dir = ais.raw_dir

all_str = "AllShips"
all_total_list = []
all_unique_list = []

tank_str = "TankerShips"
tank_total_list = []
tank_unique_list = []

pass_str = "PassengerShips"
pass_total_list = []
pass_unique_list = []

other_str = "OtherShips"
other_total_list = []
other_unique_list = []

cargo_str = "CargoShips"
cargo_total_list = []
cargo_unique_list = []

ac_ras = ac.raster_filename()

# Define Output Variables
all_total_ras = ais.raster_filename("all_t")
tank_total_ras = ais.raster_filename("tank_t")
pass_total_ras = ais.raster_filename("pass_t")
other_total_ras = ais.raster_filename("other_t")
cargo_total_ras = ais.raster_filename("cargo_t")

all_unique_ras = ais.raster_filename("all_u")
tank_unique_ras = ais.raster_filename("tank_u")
pass_unique_ras = ais.raster_filename("pass_u")
other_unique_ras = ais.raster_filename("other_u")
cargo_unique_ras = ais.raster_filename("cargo_u")

ac_all_ras = ais.raster_filename("all_ac")
all_unique_rc = ais.raster_classified_filename("a_ac_u")

tank_unique_rc = ais.raster_classified_filename("tank_u")
pass_unique_rc = ais.raster_classified_filename("pass_u")

ac_non_tank_ras = ais.raster_filename("ntk_ac")
ac_non_tank_rc = ais.raster_classified_filename("ntk_ac")

ac_non_pass_ras = ais.raster_filename("np_ac")
ac_non_pass_rc = ais.raster_classified_filename("np_ac")

tank_unique_1 = ais.raster_classified_filename("tank_1")
pass_unique_0 = ais.raster_classified_filename("pass_0")

pass_sar_rc = ais.raster_classified_filename("sar_p")  # p_len
ac_non_pass_0 = ais.raster_classified_filename("np_ac0")
npass_sar_rc = ais.raster_classified_filename("sar_np")

eez_neg5_ras = ais.working_rastername("eez_neg5")

tank_unique_rc_t = ais.working_rastername("tank_u_t")
pass_unique_rc_t = ais.working_rastername("pass_u_t")

all_unique_rf = ais.raster_final_filename("a_ac_u")
tank_unique_rf = ais.raster_final_filename("tanku")
pass_unique_rf = ais.raster_final_filename("passu")
ac_non_tank_rf = ais.raster_final_filename("ntk_ac")
ac_non_pass_rf = ais.raster_final_filename("np_ac")
pass_sar_rf = ais.raster_final_filename("sar_p")  # p_len
npass_sar_rf = ais.raster_final_filename("sar_np")

# Generate List of Filenames for Total Vessel Count (All, Pass, Tank) and Unique Vessel Count (All)
# These tif files were downloaded from http://ais.axds.co/
for root, dirs, files in os.walk(ais_raw_dir):
    for file in files:
        if file.endswith("_TotalVesselTraffic.tif"):
            if all_str in file:
                all_total_list.append(os.path.join(root, file))
            if tank_str in file:
                tank_total_list.append(os.path.join(root, file))
            if pass_str in file:
                pass_total_list.append(os.path.join(root, file))
            if other_str in file:
                other_total_list.append(os.path.join(root, file))
            if cargo_str in file:
                cargo_total_list.append(os.path.join(root, file))
        if file.endswith("_UniqueVesselTraffic.tif"):
            if all_str in file:
                all_unique_list.append(os.path.join(root, file))
            if tank_str in file:
                tank_unique_list.append(os.path.join(root, file))
            if pass_str in file:
                pass_unique_list.append(os.path.join(root, file))
            if other_str in file:
                other_unique_list.append(os.path.join(root, file))
            if cargo_str in file:
                cargo_unique_list.append(os.path.join(root, file))

# Mosaic to New Raster
arcpy.MosaicToNewRaster_management(all_total_list, os.path.dirname(all_total_ras), os.path.basename(all_total_ras), ais.projection_number, "32_BIT_SIGNED", ais.cell_size, "1", "MAXIMUM", "FIRST")
arcpy.MosaicToNewRaster_management(tank_total_list, os.path.dirname(tank_total_ras), os.path.basename(tank_total_ras), ais.projection_number, "32_BIT_SIGNED", ais.cell_size, "1", "MAXIMUM", "FIRST")
arcpy.MosaicToNewRaster_management(pass_total_list, os.path.dirname(pass_total_ras), os.path.basename(pass_total_ras), ais.projection_number, "32_BIT_SIGNED", ais.cell_size, "1", "MAXIMUM", "FIRST")
arcpy.MosaicToNewRaster_management(other_total_list, os.path.dirname(other_total_ras), os.path.basename(other_total_ras), ais.projection_number, "32_BIT_SIGNED", ais.cell_size, "1", "MAXIMUM", "FIRST")
arcpy.MosaicToNewRaster_management(cargo_total_list, os.path.dirname(cargo_total_ras), os.path.basename(cargo_total_ras), ais.projection_number, "32_BIT_SIGNED", ais.cell_size, "1", "MAXIMUM", "FIRST")

arcpy.MosaicToNewRaster_management(all_unique_list, os.path.dirname(all_unique_ras), os.path.basename(all_unique_ras), ais.projection_number, "32_BIT_SIGNED", ais.cell_size, "1", "MAXIMUM", "FIRST")
arcpy.MosaicToNewRaster_management(tank_unique_list, os.path.dirname(tank_unique_ras), os.path.basename(tank_unique_ras), ais.projection_number, "32_BIT_SIGNED", ais.cell_size, "1", "MAXIMUM", "FIRST")
arcpy.MosaicToNewRaster_management(pass_unique_list, os.path.dirname(pass_unique_ras), os.path.basename(pass_unique_ras), ais.projection_number, "32_BIT_SIGNED", ais.cell_size, "1", "MAXIMUM", "FIRST")
arcpy.MosaicToNewRaster_management(other_unique_list, os.path.dirname(other_unique_ras), os.path.basename(other_unique_ras), ais.projection_number, "32_BIT_SIGNED", ais.cell_size, "1", "MAXIMUM", "FIRST")
arcpy.MosaicToNewRaster_management(cargo_unique_list, os.path.dirname(cargo_unique_ras), os.path.basename(cargo_unique_ras), ais.projection_number, "32_BIT_SIGNED", ais.cell_size, "1", "MAXIMUM", "FIRST")

# Find maximum values of unique rasters
# Tanker - Binary (present/absent), where absent is set to NAN. To be used for clipping of Reefs/Sanctuaries dataset
all_unique_max_t = arcpy.GetRasterProperties_management(all_unique_ras, "MAXIMUM")
all_unique_max = (float(all_unique_max_t.getOutput(0))) + 100  # Ensure maximum is not mis-classified.

tank_unique_max_t = arcpy.GetRasterProperties_management(tank_unique_ras, "MAXIMUM")
tank_unique_max = (float(tank_unique_max_t.getOutput(0))) + 100  # Ensure maximum is not mis-classified.

pass_unique_max_t = arcpy.GetRasterProperties_management(pass_unique_ras, "MAXIMUM")
pass_unique_max = (float(pass_unique_max_t.getOutput(0))) + 100  # Ensure maximum is not mis-classified.

other_unique_max_t = arcpy.GetRasterProperties_management(other_unique_ras, "MAXIMUM")
other_unique_max = (float(other_unique_max_t.getOutput(0))) + 100  # Ensure maximum is not mis-classified.

cargo_unique_max_t = arcpy.GetRasterProperties_management(cargo_unique_ras, "MAXIMUM")
cargo_unique_max = (float(cargo_unique_max_t.getOutput(0))) + 100  # Ensure maximum is not mis-classified.

ac_max_t = arcpy.GetRasterProperties_management(ac_ras, "MAXIMUM")
ac_max = (float(ac_max_t.getOutput(0))) + 100  # Ensure maximum is not mis-classified.

ais_max = max([all_unique_max, ac_max])

# Reclassify
# Likelihood (AIS and AC)
arcpy.MosaicToNewRaster_management([all_unique_ras, ac_ras], os.path.dirname(ac_all_ras), os.path.basename(ac_all_ras), ais.projection_number, "32_BIT_SIGNED", ais.cell_size, "1", "MAXIMUM", "FIRST")
arcpy.gp.Reclassify_sa(ac_all_ras, "VALUE", "NODATA 1;0 0 1;1 2 2;3 10 3;11 100 4;101 %s 5" % (ais_max), all_unique_rc, "NODATA")

# Likelihood (Tanker (Reef) and Passenger (SAR)) --> Raster Final?
arcpy.gp.Reclassify_sa(tank_unique_ras, "VALUE", "NODATA 1;0 0 1;1 2 2;3 10 3;11 100 4;101 %s 5" % (ais_max), tank_unique_rc, "NODATA")
arcpy.gp.Reclassify_sa(pass_unique_ras, "VALUE", "NODATA 1;0 0 1;1 2 2;3 10 3;11 100 4;101 %s 5" % (ais_max), pass_unique_rc, "NODATA")

# Likelihood (All-But-Tanker (Reef) and All-But-Passenger (SAR))
arcpy.MosaicToNewRaster_management([cargo_unique_ras, other_unique_ras, pass_unique_ras, ac_ras], os.path.dirname(ac_non_tank_ras), os.path.basename(ac_non_tank_ras), ais.projection_number, "32_BIT_SIGNED", ais.cell_size, "1", "MAXIMUM", "FIRST")
arcpy.gp.Reclassify_sa(ac_non_tank_ras, "VALUE", "NODATA 1;0 0 1;1 2 2;3 10 3;11 100 4;101 %s 5" % (ais_max), ac_non_tank_rc, "NODATA")

arcpy.MosaicToNewRaster_management([cargo_unique_ras, other_unique_ras, tank_unique_ras, ac_ras], os.path.dirname(ac_non_pass_ras), os.path.basename(ac_non_pass_ras), ais.projection_number, "32_BIT_SIGNED", ais.cell_size, "1", "MAXIMUM", "FIRST")
arcpy.gp.Reclassify_sa(ac_non_pass_ras, "VALUE", "NODATA 1;0 0 1;1 2 2;3 10 3;11 100 4;101 %s 5" % (ais_max), ac_non_pass_rc, "NODATA")

# Tanker Reefs for Presence/Absence (Set Tanker Traffic = 1)
arcpy.gp.Reclassify_sa(tank_unique_ras, "VALUE", "1 %s 1" % (ais_max), tank_unique_1, "NODATA")

# Passenger SAR: Combine 2014 and 2016 datasets.
# For 2016 datasets, assume, where passenger vessel present, length (not currently available for 2016) is < 10 m. All passenger vessels < 10 m are classified as 0
arcpy.gp.Reclassify_sa(pass_unique_ras, "VALUE", "0 %s 0" % (ais_max), pass_unique_0, "NODATA")
arcpy.MosaicToNewRaster_management([pass_unique_0, ais_pass_len_rc_2014], os.path.dirname(pass_sar_rc), os.path.basename(pass_sar_rc), ais.projection_number, "32_BIT_SIGNED", ais.cell_size, "1", "MAXIMUM", "FIRST")

# Non-Passenger SAR: Set All-But-Passenger Dataset = 0; Mosaic with EEZ = -5 (no vessel presence)
arcpy.gp.Reclassify_sa(ac_non_pass_ras, "VALUE", "1 %s 0" % (ais_max), ac_non_pass_0, "NODATA")

eez_1_raster = arcpy.Raster(eez_1)
eez_neg4 = eez_1_raster * -5
eez_neg4.save(eez_neg5_ras)

arcpy.MosaicToNewRaster_management([ac_non_pass_0, eez_neg5_ras], os.path.dirname(npass_sar_rc), os.path.basename(npass_sar_rc), ais.projection_number, "32_BIT_SIGNED", ais.cell_size, "1", "MAXIMUM", "FIRST")

# Raster Final
# Merge Tanker and Passenger Unique with EEZ = 1 (bounds of tank/pass ais traffic are not equal to model bounds)
arcpy.MosaicToNewRaster_management([tank_unique_rc, eez_1], os.path.dirname(tank_unique_rc_t), os.path.basename(tank_unique_rc_t), ais.projection_number, "32_BIT_SIGNED", ais.cell_size, "1", "MAXIMUM", "FIRST")
arcpy.MosaicToNewRaster_management([pass_unique_rc, eez_1], os.path.dirname(pass_unique_rc_t), os.path.basename(pass_unique_rc_t), ais.projection_number, "32_BIT_SIGNED", ais.cell_size, "1", "MAXIMUM", "FIRST")


# Clip All Rasters to EEZ Bounds.
au = arcpy.sa.ExtractByMask(all_unique_rc, eez_1)
au.save(all_unique_rf)

tu = arcpy.sa.ExtractByMask(tank_unique_rc_t, eez_1)
tu.save(tank_unique_rf)

pu = arcpy.sa.ExtractByMask(pass_unique_rc_t, eez_1)
pu.save(pass_unique_rf)

ntu = arcpy.sa.ExtractByMask(ac_non_tank_rc, eez_1)
ntu.save(ac_non_tank_rf)

npu = arcpy.sa.ExtractByMask(ac_non_pass_rc, eez_1)
npu.save(ac_non_pass_rf)

psar = arcpy.sa.ExtractByMask(pass_sar_rc, eez_1)
psar.save(pass_sar_rf)

npsar = arcpy.sa.ExtractByMask(npass_sar_rc, eez_1)
npsar.save(npass_sar_rf)
