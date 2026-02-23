import os
import argparse
from HSTB.ecs import ECSObject
from HSTB.osgeo_importer import *  # sets up and imports osgeo, ogr

class S57_to_Shape:
    def __init__(self, filename):
        ftype = 'ESRI Shapefile'
        t_srs = osr.SpatialReference()
        t_srs.SetFromUserInput('WGS84')
        shp_driver = ogr.GetDriverByName(ftype)

        if os.path.exists(filename):
            os.remove(filename)
        self.shp_ds = shp_driver.CreateDataSource(str(filename))
        self.fid = 0
        if self.shp_ds is not None:
            self.shp_layer = self.shp_ds.CreateLayer('Polygon', geom_type=ogr.wkbPolygon, srs=t_srs)
            fd = ogr.FieldDefn('source', ogr.OFTString)
            fd.SetWidth(10)
            self.shp_layer.CreateField(fd)
            fd = ogr.FieldDefn('CATZOC', ogr.OFTInteger)
            self.shp_layer.CreateField(fd)
        else:
            print('failed to create', ftype, ' filename= ', filename)
            self.shp_layer = None

    def __del__(self):
        if self.shp_ds:
            self.shp_ds.Destroy()

    def add_polyline_to_shape(self, polygon, ptype, pval):
        self.fid += 1
        # line = ogr.Geometry(ogr.wkbLineString)
        # for lon, lat in s.GetPoints():
        #     line.AddPoint_2D(lon, lat)
        feat = ogr.Feature(feature_def=self.shp_layer.GetLayerDefn())
        feat.SetGeometry(polygon)
        feat.SetFID(self.fid)
        feat.SetField('source', ptype)
        feat.SetField('CATZOC', pval)
        self.shp_layer.CreateFeature(feat)
        feat.Destroy()

    def export_contour(self, feature):
        ptype = feature['SORIND'][len(feature['SORIND'])-6:]
        pval = feature['CATZOC']
        polygon = feature['ogrGeom']
        self.add_polyline_to_shape(polygon, ptype, pval)

'''
# build list of all s57 files (ones that have .000 extension).  s57_files is the list
for root, dirs, files in os.walk(mquals):
    # [f for f in files if os.path.splitext(file)[1] in ['.000']]
    for file in files:
        if os.path.splitext(file)[1] in ['.000']: # If x or y, then create list [x, y] if argument in []
             s57_files.append(os.path.join(root,file))
             count+=1
             print count

# look at all files in s57_files and return geometry and CATZOC
s57_dict = {}
for f in s57_files:
    ECSObject.ECS.LoadENCtoSENCdatabase(f)

    # grab just that cell / nested-dictionary that is of ECSObject
    dictENCcell = ECSObject.ECS.ENCcellFeatures[f]
    
    # assign feature generator per the M_QUAL-only filter
    fg = ECSObject.iterENCObjects(dictENCcell,filter_to_M_QUALonly)

    # iterate on that to get [the one?] M_QUAL
    for vDPRI,geomType,sIMO,vOGRUP,cRPRI,sENCgroup,oAcronym,features in fg:
        for featureID,feature in features.iteritems():
            hnum = feature['SORIND'][len(feature['SORIND'])-6:]
            s57_dict[hnum] = [feature['CATZOC'], feature['ogrGeom']]
'''
#more efficient to combine these (otherwise you iterate through 3000 files twice
def return_catzoc_geom(mquals):
    s57_files = {}

    filter_to_M_QUALonly = {'fontPtSizeSpinCtrl': 10, 'safcntKey': 'safcnt7', 'natsurAbbrTextCheckbox': True, u'featSymbSizeM': 100.0, 'haloTextCheckbox': False, 'fontNameChoice': 'Arial', 's52SafetyContourUnitsChoice': 'meters', 'geomENC': {'Points': True, 'Lines': True, 'Areas': True}, 'clipBSB': True, 'chartfilterTheme': 'S-57 Object Classes', 'cRPRI': {'S': True, 'O': True}, 'geomObj': {u'Points': {'$': ['$COMPS', '$CSYMB', '$TEXTS'], 'G': ['ACHARE', 'ACHBRT', 'AIRARE', 'BCNCAR', 'BCNISD', 'BCNLAT', 'BCNSAW', 'BCNSPP', 'BERTHS', 'BOYCAR', 'BOYINB', 'BOYISD', 'BOYLAT', 'BOYSAW', 'BOYSPP', 'BRIDGE', 'BUAARE', 'BUISGL', 'CGUSTA', 'CHKPNT', 'CRANES', 'CTNARE', 'CTRPNT', 'CTSARE', 'CURENT', 'DAMCON', 'DAYMAR', 'DISMAR', 'DMPGRD', 'FOGSIG', 'FORSTC', 'FSHFAC', 'GATCON', 'GRIDRN', 'HRBFAC', 'HULKES', 'ICNARE', 'LIGHTS', 'LITFLT', 'LITVES', 'LNDARE', 'LNDELV', 'LNDMRK', 'LNDRGN', 'LOCMAG', 'LOGPON', 'MAGVAR', 'MARCUL', 'MIPARE', 'MORFAC', 'NEWOBJ', 'OBSTRN', 'OFSPLF', 'PILBOP', 'PILPNT', 'PIPARE', 'PIPSOL', 'PRCARE', 'PRDARE', 'PYLONS', 'RADRFL', 'RADSTA', 'RAPIDS', 'RCTLPT', 'RDOCAL', 'RDOSTA', 'RETRFL', 'ROADWY', 'RSCSTA', 'RTPBCN', 'RUNWAY', 'SBDARE', 'SEAARE', 'SILTNK', 'SISTAT', 'SISTAW', 'SLCONS', 'SLOGRD', 'SMCFAC', 'SNDWAV', 'SOUNDG', 'SPLARE', 'SPRING', 'SQUARE', 'TOPMAR', 'TS_FEB', 'TS_PAD', 'TS_PNH', 'TS_PRH', 'TS_TIS', 'TUNNEL', 'T_HMON', 'T_NHMN', 'T_TIMS', 'UWTROC', 'VEGATN', 'WATFAL', 'WATTUR', 'WEDKLP', 'WRECKS', 'usrmrk']}, 'Cand$': ['C_AGGR', 'C_ASSO', 'C_STAC'], u'Lines': {'$': ['$LINES'], 'G': ['ASLXIS', 'BERTHS', 'BRIDGE', 'CANALS', 'CANBNK', 'CAUSWY', 'CBLOHD', 'CBLSUB', 'COALNE', 'CONVYR', 'DAMCON', 'DEPARE', 'DEPCNT', 'DWRTCL', 'DYKCON', 'FERYRT', 'FLODOC', 'FNCLNE', 'FORSTC', 'FSHFAC', 'GATCON', 'LAKSHR', 'LNDARE', 'LNDELV', 'LNDMRK', 'LOCMAG', 'MAGVAR', 'MARCUL', 'MORFAC', 'NAVLNE', 'NEWOBJ', 'OBSTRN', 'OILBAR', 'PIPOHD', 'PIPSOL', 'PONTON', 'RADLNE', 'RAILWY', 'RAPIDS', 'RCRTCL', 'RDOCAL', 'RECTRC', 'RIVBNK', 'RIVERS', 'ROADWY', 'RUNWAY', 'SBDARE', 'SLCONS', 'SLOTOP', 'SNDWAV', 'SQUARE', 'STSLNE', 'TIDEWY', 'TSELNE', 'TSSBND', 'TUNNEL', 'VEGATN', 'WATFAL', 'WATTUR', 'brklne', 'usrmrk']}, 'mode': {'C': 'None', 'M': 'Select', '$': 'None', 'G': 'None'}, u'Areas': {'M': ['M_QUAL'], '$': ['$AREAS'], 'G': ['ACHARE', 'ACHBRT', 'ADMARE', 'AIRARE', 'ARCSLN', 'BERTHS', 'BRIDGE', 'BUAARE', 'BUISGL', 'CANALS', 'CANBNK', 'CAUSWY', 'CBLARE', 'CHKPNT', 'CONVYR', 'CONZNE', 'COSARE', 'CRANES', 'CTNARE', 'CTSARE', 'CUSZNE', 'DAMCON', 'DEPARE', 'DMPGRD', 'DOCARE', 'DRGARE', 'DRYDOC', 'DWRTPT', 'DYKCON', 'EXEZNE', 'FAIRWY', 'FERYRT', 'FLODOC', 'FORSTC', 'FRPARE', 'FSHFAC', 'FSHGRD', 'FSHZNE', 'GATCON', 'GRIDRN', 'HRBARE', 'HRBFAC', 'HULKES', 'ICEARE', 'ICNARE', 'ISTZNE', 'LAKARE', 'LAKSHR', 'LNDARE', 'LNDMRK', 'LNDRGN', 'LOCMAG', 'LOGPON', 'LOKBSN', 'MAGVAR', 'MARCUL', 'MIPARE', 'MORFAC', 'NEWOBJ', 'OBSTRN', 'OFSPLF', 'OSPARE', 'PILBOP', 'PIPARE', 'PONTON', 'PRCARE', 'PRDARE', 'PYLONS', 'RADRNG', 'RAPIDS', 'RCTLPT', 'RECTRC', 'RESARE', 'RIVBNK', 'RIVERS', 'ROADWY', 'RUNWAY', 'SBDARE', 'SEAARE', 'SILTNK', 'SLCONS', 'SLOGRD', 'SMCFAC', 'SNDWAV', 'SPLARE', 'SQUARE', 'SUBTLN', 'SWPARE', 'TESARE', 'TIDEWY', 'TSEZNE', 'TSSCRS', 'TSSLPT', 'TSSRON', 'TS_FEB', 'TS_PAD', 'TS_PNH', 'TS_PRH', 'TS_TIS', 'TUNNEL', 'TWRTPT', 'T_HMON', 'T_NHMN', 'T_TIMS', 'UNSARE', 'VEGATN', 'WATTUR', 'WEDKLP', 'WRECKS', 'brklne', 'cvrage', 'm_pyup', 'prodpf', 'surfac', 'survey', 'usrmrk']}}, 'sENCgroup': [None, True, True], 's52ShallowContourUnitsChoice': 'meters', 'encAreaObjSymbChoice': 'Plain Boundaries', 'encPointObjSymbChoice': 'Simplified ECDIS', 's52DeepContourUnitsChoice': 'meters', 's52SafetyDepthUnitsChoice': 'meters', 'vDPRI': [True, True, True, True, True, True, True, True, True, True], 'featattr': {'textGroups': {'modeOther': 'None', 'modeImportant': 'None', 'vTOGRUP': [10, 11, 20, 21, 23, 24, 25, 26, 27, 28, 29, 31]}, 'DATSTA/END': False, 'PERSTA/END': False, 'SCAMIN': False, 'chart1Theme': [True, True, True, True, True, True, True, True, True], 'SCAMAX': False}, 'swisodTextCheckbox': False, 'sIMO': {'DISPLAYBASE': {'mode': 'None', 'vOGRUP': []}, 'OTHER': {'mode': 'Select', 'vOGRUP': []}, 'STANDARD': {'mode': 'All', 'vOGRUP': [302]}}, 's52ColorSchemeChoice': 'Day Bright'}
    shapefile = S57_to_Shape(r'C:\Users\Christina.Fandel\Documents\output.shp')
    
    for root, dirs, files in os.walk(mquals):
        # [f for f in files if os.path.splitext(file)[1] in ['.000']]
        for count, file in enumerate(files):
            if os.path.splitext(file)[1] in ['.000'] and "_CS." in file: # If x or y, then create list [x, y] if argument in []
                print('found {}'.format(file))
                filename = os.path.join(root, file)
                try:
                    ECSObject.ECS.LoadENCtoSENCdatabase(filename)
                    try:
                        # grab just that cell / nested-dictionary that is of ECSObject
                        dictENCcell = ECSObject.ECS.ENCcellFeatures[filename]
                        
                        # assign feature generator per the M_QUAL-only filter
                        fg = ECSObject.iterENCObjects(dictENCcell,filter_to_M_QUALonly)
            
                        # iterate on that to get [the one?] M_QUAL
                        for vDPRI,geomType,sIMO,vOGRUP,cRPRI,sENCgroup,oAcronym,features in fg:
                            for featureID,feature in features.iteritems():
                                hnum = feature['SORIND'][len(feature['SORIND'])-6:]
                                try:
                                    shapefile.export_contour(feature)
                                except:
                                    print('***FAILED - export contour {}***'.format(file))
                                    pass
                                try:
                                    s57_files[hnum] = [feature['CATZOC'], feature['ogrGeom']]
                                except:
                                    print('***FAILED - append s57 list {}***'.format(file))
                                    pass
                        print(count)
                    except:
                        print("FAILED {}".format(file))
                    finally:
                        ECSObject.ECS.PurgeSENCdatabase(filename)
                except:
                    print("FAILED to load {}".format(file))
    return s57_files

if __name__ == "__main__":
    return_catzoc_geom(r'Z:\Products\Data Archive')
    
    
    


    
    
    



'''import os
import argparse
from HSTB.ecs import ECSObject
from HSTB.osgeo_importer import *  # sets up and imports osgeo, ogr

class S57_to_Shape:
    def __init__(self, filename):
        ftype = 'ESRI Shapefile'
        t_srs = osr.SpatialReference()
        t_srs.SetFromUserInput('WGS84')
        shp_driver = ogr.GetDriverByName(ftype)

        if os.path.exists(filename):
            os.remove(filename)
        self.shp_ds = shp_driver.CreateDataSource(str(filename))
        self.fid = 0
        if self.shp_ds is not None:
            self.shp_layer = self.shp_ds.CreateLayer('Polygon', geom_type=ogr.wkbPolygon, srs=t_srs)
            fd = ogr.FieldDefn('source', ogr.OFTString)
            fd.SetWidth(10)
            self.shp_layer.CreateField(fd)
            fd = ogr.FieldDefn('CATZOC', ogr.OFTInteger)
            self.shp_layer.CreateField(fd)
        else:
            print('failed to create', ftype, ' filename= ', filename)
            self.shp_layer = None

    def __del__(self):
        if self.shp_ds:
            self.shp_ds.Destroy()

    def add_polyline_to_shape(self, polygon, ptype, pval):
        self.fid += 1
        # line = ogr.Geometry(ogr.wkbLineString)
        # for lon, lat in s.GetPoints():
        #     line.AddPoint_2D(lon, lat)
        feat = ogr.Feature(feature_def=self.shp_layer.GetLayerDefn())
        feat.SetGeometry(polygon)
        feat.SetFID(self.fid)
        feat.SetField('source', ptype)
        feat.SetField('CATZOC', pval)
        self.shp_layer.CreateFeature(feat)
        feat.Destroy()

    def export_contour(self, feature):
        ptype = feature['SORIND'][len(feature['SORIND'])-6:]
        pval = feature['CATZOC']
        polygon = feature['ogrGeom']
        self.add_polyline_to_shape(polygon, ptype, pval)

'''
# build list of all s57 files (ones that have .000 extension).  s57_files is the list
for root, dirs, files in os.walk(mquals):
    # [f for f in files if os.path.splitext(file)[1] in ['.000']]
    for file in files:
        if os.path.splitext(file)[1] in ['.000']: # If x or y, then create list [x, y] if argument in []
             s57_files.append(os.path.join(root,file))
             count+=1
             print count

# look at all files in s57_files and return geometry and CATZOC
s57_dict = {}
for f in s57_files:
    ECSObject.ECS.LoadENCtoSENCdatabase(f)

    # grab just that cell / nested-dictionary that is of ECSObject
    dictENCcell = ECSObject.ECS.ENCcellFeatures[f]
    
    # assign feature generator per the M_QUAL-only filter
    fg = ECSObject.iterENCObjects(dictENCcell,filter_to_M_QUALonly)

    # iterate on that to get [the one?] M_QUAL
    for vDPRI,geomType,sIMO,vOGRUP,cRPRI,sENCgroup,oAcronym,features in fg:
        for featureID,feature in features.iteritems():
            hnum = feature['SORIND'][len(feature['SORIND'])-6:]
            s57_dict[hnum] = [feature['CATZOC'], feature['ogrGeom']]

#more efficient to combine these (otherwise you iterate through 3000 files twice
def return_catzoc_geom(mquals):
    s57_files = {}
    count=0
    filter_to_M_QUALonly = {'fontPtSizeSpinCtrl': 10, 'safcntKey': 'safcnt7', 'natsurAbbrTextCheckbox': True, u'featSymbSizeM': 100.0, 'haloTextCheckbox': False, 'fontNameChoice': 'Arial', 's52SafetyContourUnitsChoice': 'meters', 'geomENC': {'Points': True, 'Lines': True, 'Areas': True}, 'clipBSB': True, 'chartfilterTheme': 'S-57 Object Classes', 'cRPRI': {'S': True, 'O': True}, 'geomObj': {u'Points': {'$': ['$COMPS', '$CSYMB', '$TEXTS'], 'G': ['ACHARE', 'ACHBRT', 'AIRARE', 'BCNCAR', 'BCNISD', 'BCNLAT', 'BCNSAW', 'BCNSPP', 'BERTHS', 'BOYCAR', 'BOYINB', 'BOYISD', 'BOYLAT', 'BOYSAW', 'BOYSPP', 'BRIDGE', 'BUAARE', 'BUISGL', 'CGUSTA', 'CHKPNT', 'CRANES', 'CTNARE', 'CTRPNT', 'CTSARE', 'CURENT', 'DAMCON', 'DAYMAR', 'DISMAR', 'DMPGRD', 'FOGSIG', 'FORSTC', 'FSHFAC', 'GATCON', 'GRIDRN', 'HRBFAC', 'HULKES', 'ICNARE', 'LIGHTS', 'LITFLT', 'LITVES', 'LNDARE', 'LNDELV', 'LNDMRK', 'LNDRGN', 'LOCMAG', 'LOGPON', 'MAGVAR', 'MARCUL', 'MIPARE', 'MORFAC', 'NEWOBJ', 'OBSTRN', 'OFSPLF', 'PILBOP', 'PILPNT', 'PIPARE', 'PIPSOL', 'PRCARE', 'PRDARE', 'PYLONS', 'RADRFL', 'RADSTA', 'RAPIDS', 'RCTLPT', 'RDOCAL', 'RDOSTA', 'RETRFL', 'ROADWY', 'RSCSTA', 'RTPBCN', 'RUNWAY', 'SBDARE', 'SEAARE', 'SILTNK', 'SISTAT', 'SISTAW', 'SLCONS', 'SLOGRD', 'SMCFAC', 'SNDWAV', 'SOUNDG', 'SPLARE', 'SPRING', 'SQUARE', 'TOPMAR', 'TS_FEB', 'TS_PAD', 'TS_PNH', 'TS_PRH', 'TS_TIS', 'TUNNEL', 'T_HMON', 'T_NHMN', 'T_TIMS', 'UWTROC', 'VEGATN', 'WATFAL', 'WATTUR', 'WEDKLP', 'WRECKS', 'usrmrk']}, 'Cand$': ['C_AGGR', 'C_ASSO', 'C_STAC'], u'Lines': {'$': ['$LINES'], 'G': ['ASLXIS', 'BERTHS', 'BRIDGE', 'CANALS', 'CANBNK', 'CAUSWY', 'CBLOHD', 'CBLSUB', 'COALNE', 'CONVYR', 'DAMCON', 'DEPARE', 'DEPCNT', 'DWRTCL', 'DYKCON', 'FERYRT', 'FLODOC', 'FNCLNE', 'FORSTC', 'FSHFAC', 'GATCON', 'LAKSHR', 'LNDARE', 'LNDELV', 'LNDMRK', 'LOCMAG', 'MAGVAR', 'MARCUL', 'MORFAC', 'NAVLNE', 'NEWOBJ', 'OBSTRN', 'OILBAR', 'PIPOHD', 'PIPSOL', 'PONTON', 'RADLNE', 'RAILWY', 'RAPIDS', 'RCRTCL', 'RDOCAL', 'RECTRC', 'RIVBNK', 'RIVERS', 'ROADWY', 'RUNWAY', 'SBDARE', 'SLCONS', 'SLOTOP', 'SNDWAV', 'SQUARE', 'STSLNE', 'TIDEWY', 'TSELNE', 'TSSBND', 'TUNNEL', 'VEGATN', 'WATFAL', 'WATTUR', 'brklne', 'usrmrk']}, 'mode': {'C': 'None', 'M': 'Select', '$': 'None', 'G': 'None'}, u'Areas': {'M': ['M_QUAL'], '$': ['$AREAS'], 'G': ['ACHARE', 'ACHBRT', 'ADMARE', 'AIRARE', 'ARCSLN', 'BERTHS', 'BRIDGE', 'BUAARE', 'BUISGL', 'CANALS', 'CANBNK', 'CAUSWY', 'CBLARE', 'CHKPNT', 'CONVYR', 'CONZNE', 'COSARE', 'CRANES', 'CTNARE', 'CTSARE', 'CUSZNE', 'DAMCON', 'DEPARE', 'DMPGRD', 'DOCARE', 'DRGARE', 'DRYDOC', 'DWRTPT', 'DYKCON', 'EXEZNE', 'FAIRWY', 'FERYRT', 'FLODOC', 'FORSTC', 'FRPARE', 'FSHFAC', 'FSHGRD', 'FSHZNE', 'GATCON', 'GRIDRN', 'HRBARE', 'HRBFAC', 'HULKES', 'ICEARE', 'ICNARE', 'ISTZNE', 'LAKARE', 'LAKSHR', 'LNDARE', 'LNDMRK', 'LNDRGN', 'LOCMAG', 'LOGPON', 'LOKBSN', 'MAGVAR', 'MARCUL', 'MIPARE', 'MORFAC', 'NEWOBJ', 'OBSTRN', 'OFSPLF', 'OSPARE', 'PILBOP', 'PIPARE', 'PONTON', 'PRCARE', 'PRDARE', 'PYLONS', 'RADRNG', 'RAPIDS', 'RCTLPT', 'RECTRC', 'RESARE', 'RIVBNK', 'RIVERS', 'ROADWY', 'RUNWAY', 'SBDARE', 'SEAARE', 'SILTNK', 'SLCONS', 'SLOGRD', 'SMCFAC', 'SNDWAV', 'SPLARE', 'SQUARE', 'SUBTLN', 'SWPARE', 'TESARE', 'TIDEWY', 'TSEZNE', 'TSSCRS', 'TSSLPT', 'TSSRON', 'TS_FEB', 'TS_PAD', 'TS_PNH', 'TS_PRH', 'TS_TIS', 'TUNNEL', 'TWRTPT', 'T_HMON', 'T_NHMN', 'T_TIMS', 'UNSARE', 'VEGATN', 'WATTUR', 'WEDKLP', 'WRECKS', 'brklne', 'cvrage', 'm_pyup', 'prodpf', 'surfac', 'survey', 'usrmrk']}}, 'sENCgroup': [None, True, True], 's52ShallowContourUnitsChoice': 'meters', 'encAreaObjSymbChoice': 'Plain Boundaries', 'encPointObjSymbChoice': 'Simplified ECDIS', 's52DeepContourUnitsChoice': 'meters', 's52SafetyDepthUnitsChoice': 'meters', 'vDPRI': [True, True, True, True, True, True, True, True, True, True], 'featattr': {'textGroups': {'modeOther': 'None', 'modeImportant': 'None', 'vTOGRUP': [10, 11, 20, 21, 23, 24, 25, 26, 27, 28, 29, 31]}, 'DATSTA/END': False, 'PERSTA/END': False, 'SCAMIN': False, 'chart1Theme': [True, True, True, True, True, True, True, True, True], 'SCAMAX': False}, 'swisodTextCheckbox': False, 'sIMO': {'DISPLAYBASE': {'mode': 'None', 'vOGRUP': []}, 'OTHER': {'mode': 'Select', 'vOGRUP': []}, 'STANDARD': {'mode': 'All', 'vOGRUP': [302]}}, 's52ColorSchemeChoice': 'Day Bright'}
    shapefile = S57_to_Shape(r'C:\Users\Christina.Fandel\Documents\output.shp')
    
    for root, dirs, files in os.walk(mquals):
        # [f for f in files if os.path.splitext(file)[1] in ['.000']]
        for file in files:
            if os.path.splitext(file)[1] in ['.000']: # If x or y, then create list [x, y] if argument in []
                print('found')
                filename = os.path.join(root, file)
                ECSObject.ECS.LoadENCtoSENCdatabase(filename)

                # grab just that cell / nested-dictionary that is of ECSObject
                dictENCcell = ECSObject.ECS.ENCcellFeatures[filename]
                
                # assign feature generator per the M_QUAL-only filter
                fg = ECSObject.iterENCObjects(dictENCcell,filter_to_M_QUALonly)

                # iterate on that to get [the one?] M_QUAL
                for vDPRI,geomType,sIMO,vOGRUP,cRPRI,sENCgroup,oAcronym,features in fg:
                    for featureID,feature in features.iteritems():
                        hnum = feature['SORIND'][len(feature['SORIND'])-6:]
                        try:
                            shapefile.export_contour(feature)
                        except:
                            print('***FAILED - export contour {}***'.format(file))
                            pass
                        try:
                            s57_files[hnum] = [feature['CATZOC'], feature['ogrGeom']]
                        except:
                            print('***FAILED - append s57 list {}***'.format(file))
                            pass
                count+=1
                print(count)
    return s57_files

if __name__ == "__main__":
    return_catzoc_geom(r'Z:\Products\Data Archive')

307
found
Traceback (most recent call last):
  File "C:\PYDRO6~2\NOAA\site-packages\Python2\HSTB\ecs\ECSObject.py", line 808, in OpenENCObject
    if ogrENCcell.GetDriver().GetName().upper()!="S57":
AttributeError: 'NoneType' object has no attribute 'GetDriver'
Traceback (most recent call last):
  File "C:\PYDRO6~2\NOAA\site-packages\Python2\HSTB\demos\HowTo_Make_A_Simple_Demo.py.mod.py", line 118, in <module>
    return_catzoc_geom(r'Z:\Products\Data Archive')
  File "C:\PYDRO6~2\NOAA\site-packages\Python2\HSTB\demos\HowTo_Make_A_Simple_Demo.py.mod.py", line 91, in return_catzoc_geom
    ECSObject.ECS.LoadENCtoSENCdatabase(filename)
  File "C:\PYDRO6~2\NOAA\site-packages\Python2\HSTB\ecs\ECSObject.py", line 1211, in LoadENCtoSENCdatabase
    if self.OpenENCObject(path2ENCBaseCell,clipping):
  File "C:\PYDRO6~2\NOAA\site-packages\Python2\HSTB\ecs\ECSObject.py", line 818, in OpenENCObject
    "Error", wx.OK | wx.CENTRE | wx.ICON_EXCLAMATION, self.frame)
  File "C:\PYDRO6~2\pkgs\FromWheels\py27\extracted_wheels\wx30\wx-3.0-msw\wx\_misc.py", line 491, in MessageBox
    return _misc_.MessageBox(*args, **kwargs)
wx._core.PyNoAppError: The wx.App object must be created first!
Exception AttributeError: AttributeError("'NoneType' object has no attribute 'Error_GetErrorCount'",) in <bound method IndexStreamHandle.__del__ of <rtree.index.IndexStreamHandle object at 0x00000001B15373C8>> ignored
