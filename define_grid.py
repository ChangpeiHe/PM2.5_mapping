import imp
import numpy as np
import geopandas as gpd
import math
import rasterio
import os
from rasterio.transform import from_origin
import pandas as pd
from rasterio.mask import mask

class Grid_define:
    '''
    define spatial extent, resolution, and grid
    res: the spatial resolution of grid
    shapefile: the shapefile of interesting study region (.shp)
    '''

    extent = (-180, -90, 180, 90)
    tmp_dir = "/WORK/genggn_work/hechangpei/PM2.5/"
    crs = 'EPSG:4326'

    def __init__(self, res, shapefile=None):
        self.res = res
        self.shapefile = shapefile
        if self.shapefile is None:
            self.extent = self.extent
            self.lon_west, self.lat_south, self.lon_east, self.lat_north = self.extent
            self.col_west = self.lon_to_col(self.lon_west)
            self.col_east = self.lon_to_col(self.lon_east)
            self.row_north = self.lat_to_row(self.lat_north)
            self.row_south = self.lat_to_row(self.lat_south)
        else:
            map_shapefile = gpd.read_file(self.shapefile)
            map_shapefile = map_shapefile.to_crs(self.crs)
            self.map_shapefile = map_shapefile
            self.extent = map_shapefile.total_bounds
            self.lon_west, self.lat_south, self.lon_east, self.lat_north = self.extent
            self.col_west = self.lon_to_col(self.lon_west)-1
            self.col_east = self.lon_to_col(self.lon_east)+1
            self.row_north = self.lat_to_row(self.lat_north)-1
            self.row_south = self.lat_to_row(self.lat_south)+1
        self.row_list = list(range(self.col_west, self.col_east+1, 1))
        self.col_list = list(range(self.row_north, self.row_south+1, 1))

    def lon_to_col(self, lon):
        col = math.ceil((lon+180)/self.res)
        return col

    def lat_to_row(self, lat):
        row = math.ceil((90-lat)/self.res) 
        return row
    
    def col_to_lon(self, col):
        lon = self.res/2+(col-1)*self.res-180
        return lon

    def row_to_lat(self, row):
        lat = 90-self.res/2-(row-1)*self.res
        return lat
    
    @property
    def model_grid(self):
        if self.shapefile is None:
            self.grid_col = [self.col_to_lon(lon) for lon in grid_lon]
            self.grid_row = [self.lat_to_row(lat) for lat in grid_lat]
            df_grid = pd.DataFrame({'row': self.grid_row, 'col': self.grid_col, 'lon': self.grid_lon, 'lat': self.grid_lat})
            return(df_grid)
        else:
            x = [self.col_to_lon(col) for col in self.col_list]
            y = [self.row_to_lat(row) for row in self.row_list]
            grid_x, grid_y = np.meshgrid(x, y)
            transform = from_origin(np.min(x)-self.res/2, np.max(y)+self.res/2, self.res, self.res)
            with rasterio.open(
                os.path.join(self.tmp_dir, 'tmp.tif'), 'w', driver='GTiff', height=len(y), width=len(x), count=1, dtype='float32', crs=self.crs, transform=transform) as f:
                    f.write(np.ones((len(y), len(x))), 1)
            with rasterio.open(os.path.join(self.tmp_dir, 'tmp.tif')) as f:  
                model_grid_mask = mask(f, self.map_shapefile.geometry, crop=False, nodata=np.nan)[0]
                NA_index = np.isnan(model_grid_mask[0, :, :])
                self.grid_lon = list(grid_x[~NA_index])
                self.grid_lat = list(grid_y[~NA_index])
                self.grid_col = [self.lon_to_col(lon) for lon in self.grid_lon]
                self.grid_row = [self.lat_to_row(lat) for lat in self.grid_lat]
            df_grid = pd.DataFrame({'row': self.grid_row, 'col': self.grid_col, 'lon': self.grid_lon, 'lat': self.grid_lat})
        return(df_grid)
        

if __name__ == "__main__":	
    NA_grid = Grid_define(res=0.1, shapefile="/WORK/genggn_work/hechangpei/PM2.5/politicalboundaries_shapefile/boundaries_p_2021_v3.shp")
    df_grid = NA_grid.model_grid
    print(df_grid)


