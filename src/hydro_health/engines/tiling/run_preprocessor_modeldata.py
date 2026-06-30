import pathlib
import sys

HYDRO_HEALTH = pathlib.Path(__file__).parents[3]
sys.path.append(str(HYDRO_HEALTH))

from hydro_health.engines.tiling.ModelDataPreProcessor import ModelDataPreProcessor

if __name__ == '__main__':
    processor = ModelDataPreProcessor(overwrite=True, pilot_mode=True)
    
    print("Starting processing. (Note: Dask performance_report must be triggered inside ModelDataPreProcessor.py)")
    
    processor.process()
    
    print("Processing complete!")
    