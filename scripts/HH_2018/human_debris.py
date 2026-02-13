import os

from HSTB.ArcExt.NHSP import Functions


features_1 = ['.obstrn', '.pilpnt', '.wrecks']  # UPDATE ## Is this correct? Check Mike's notes
geoms_1 = ['_point']


def extract_raw(params):
    # RAW DATA
    # Extract Raw Data: OBSTRN, PILPNT, and WRECKS Features
    return Functions.extract_enc_data_p(params, features_1, geoms_1)
