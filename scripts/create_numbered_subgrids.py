import geopandas as gpd
import pathlib


INPUTS = pathlib.Path(__file__).parents[1] / 'inputs'
MASTER_GRIDS = INPUTS / 'Master_Grids.gpkg'


def create_tile_ids(blue_topo_tile, sub_grid_within, model_sub_grid):
    bt_tl, bt_bl, bt_br, bt_tr, _ = blue_topo_tile.geometry.geoms[0].exterior.coords
    for index, sub_grid_row in sub_grid_within.iterrows():
        tl, bl, br, tr, __ = sub_grid_row.geometry.geoms[0].exterior.coords
        if bt_tl == tl:
            tile_id = 1
        elif bt_tr == tr:
            tile_id = 2
        elif bt_br == br:
            tile_id = 3
        elif bt_bl == bl:
            tile_id = 4
        model_sub_grid.at[index, 'Tile_ID'] = f"{blue_topo_tile['Tilename']}_{tile_id}"
        

def process():
    blue_topo_grid = gpd.read_file(MASTER_GRIDS, layer='Blue_topo_Grid_Tiles')
    model_sub_grid = gpd.read_file(MASTER_GRIDS, layer='Model_sub_Grid_Tiles')

    for _, blue_row in blue_topo_grid.iterrows():
        sub_grid_within = model_sub_grid[model_sub_grid.within(blue_row.geometry)]
        create_tile_ids(blue_row, sub_grid_within, model_sub_grid)
    print(model_sub_grid.head()['Tile_ID'])
    
    model_sub_grid.to_file(MASTER_GRIDS, layer='Model_sub_Grid_Tiles', driver='GPKG')


if __name__ == "__main__":
    process()
    print('Done')