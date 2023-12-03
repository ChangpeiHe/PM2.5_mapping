import os
import numpy as np
import pandas as pd
import rasterio
import rioxarray as rxr
import geopandas as gpd
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

class Preprocess_data:
    '''
        resample all variables to predefined grid
    '''
    
    def __init__(self, grid_obj) -> None:
        self.grid_obj = grid_obj
    
    def process_pm25(self, data_dir):
        df = pd.read_csv(data_dir)
        df['Date'] = pd.to_datetime(df['Year'].astype(str) + df['DOY'].astype(str), format='%Y%j')
        df['row'] = [self.grid_obj.lat_to_row(lat) for lat in df['Lat']]
        df['col'] = [self.grid_obj.lon_to_col(lon) for lon in df['Lon']]
        df = df.groupby(['row', 'col', 'Date'])[['PM25']].mean().reset_index()
        return df
    
    def process_pop(self, data_dir):
        df_grid = self.grid_obj.model_grid
        df_rxr = rxr.open_rasterio(data_dir).sel(y=slice(self.grid_obj.lat_north+self.grid_obj.res, self.grid_obj.lat_south-self.grid_obj.res), x=slice(self.grid_obj.lon_west-self.grid_obj.res, self.grid_obj.lon_east+self.grid_obj.res))
        df = df_rxr.values[0,]
        df[df == df_rxr._FillValue] = np.nan
        x = df_rxr['x'].values
        y = df_rxr['y'].values
        grid_x, grid_y = np.meshgrid(x, y)
        known_points = np.array([list(grid_x[np.logical_not(np.isnan(df))]), list(grid_y[np.logical_not(np.isnan(df))]), list(df[np.logical_not(np.isnan(df))])]).T
        df = griddata((known_points[:,0], known_points[:,1]), known_points[:,2], (df_grid['lon'], df_grid['lat']), method='linear')
        df[np.isnan(df)] = np.nanmean(df)
        return(df)
    
    

        

