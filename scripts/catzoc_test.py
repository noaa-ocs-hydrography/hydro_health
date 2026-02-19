import pathlib
import xml.etree.ElementTree as ET
import catzoc_score

from datetime import datetime, date


INPUTS = pathlib.Path(__file__).parents[1] / "inputs"


def parse_gdal_rat_to_metadata(root):
    
    # Find the RAT and Field Definitions
    rat_node = root.find(".//GDALRasterAttributeTable")
    field_names = [f.find('Name').text for f in rat_node.findall('FieldDefn')]
    
    metadata_list = []
    
    for row in rat_node.findall('Row'):
        # Map values to their field names
        row_data = {field_names[i]: f_val.text for i, f_val in enumerate(row.findall('F'))}
        
        # Transform to your score.py requirements
        meta = {
            'from_filename': row_data.get('source_survey_id'),
            'feat_detect': bool(int(row_data.get('significant_features', 0))),
            'feat_least_depth': bool(int(row_data.get('feature_least_depth', 0))),
            'complete_coverage': bool(int(row_data.get('bathy_coverage', 0))),
            'horiz_uncert_fixed': float(row_data.get('horizontal_uncert_fixed', 0)),
            'horiz_uncert_vari': float(row_data.get('horizontal_uncert_var', 0)),
            'vert_uncert_fixed': float(row_data.get('vertical_uncert_fixed', 0)),
            'vert_uncert_vari': float(row_data.get('vertical_uncert_var', 0)),
            'start_date': datetime.strptime(row_data.get('survey_date_start'), '%Y-%m-%d').date(),
            'end_date': datetime.strptime(row_data.get('survey_date_end'), '%Y-%m-%d').date(),
            'interpolated': ".interpolated" in row_data.get('source_survey_id', '').lower()
        }
        
        # Spike's extra trick: adding the feat_size if it exists
        if 'feature_size' in row_data:
            meta['feat_size'] = float(row_data['feature_size'])
            
        metadata_list.append(meta)
        
    return metadata_list

# Test it out!
xml_string = ET.parse(INPUTS / 'test_xml_files' / 'BlueTopo_BF2GX2KK_20241009.tiff.aux.xml')
metadata_dicts = parse_gdal_rat_to_metadata(xml_string)
print(metadata_dicts[0])



# Let's say metadata_dicts is the list we got from the parser
for meta in metadata_dicts:
    try:
        # 1. Calculate the raw Supersession Score
        ss_score = catzoc_score.supersession(meta)
        meta['supersession_score'] = ss_score
        
        # 2. Determine the CATZOC (A1 through U)
        cz_value = catzoc_score.catzoc(meta)
        
        # 3. Calculate Decay (as of today)
        today = date.today()
        decayed_ss = catzoc_score.decay(meta, today)
        
        print(f"--- Survey: {meta['from_filename']} ---")
        print(f"  Raw Score: {ss_score:.2f}")
        print(f"  Decayed Score: {decayed_ss:.2f}")
        print(f"  CATZOC Category: {cz_value}")
        
    except ValueError as e:
        print(f"Mischief managed! Error scoring {meta.get('from_filename')}: {e}")