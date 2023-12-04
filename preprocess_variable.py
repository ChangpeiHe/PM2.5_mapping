import os
import numpy as np
import pandas as pd
import rasterio
import rioxarray as rxr
import geopandas as gpd
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import pyproj

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
        return df
    
    def process_burn(self, data_dir):
        month_start = pd.to_datetime(self.grid_obj.start_day, format='%Y-%m-%d').month
        month_end = pd.to_datetime(self.grid_obj.end_day, format='%Y-%m-%d').month
        df_grid = self.grid_obj.model_grid
        file_list = sorted([file for file in os.listdir(data_dir) if ('hdf' in file) and ('xml' not in file)])
        month_index = ['001', '032', '060', '091', '121', '152', '182', '213', '244', '274', '305', '335']
        month = list(range(1, 13))
        month_dict = dict(zip(month, month_index))
        df_burn_all = []
        for month in list(range(month_start, month_end+1)):
            # month = 1
            file_month_list = sorted([file for file in file_list if (str(month_dict[month])==file[13:16])])
            df_burn = []
            for i in range(len(file_month_list)):
                # i=200
                file_example = file_month_list[i]
                hdf_ori = rxr.open_rasterio(os.path.join(data_dir, file_example))
                Burn_day = hdf_ori['Burn Date'].values[0,]
                first_day = hdf_ori['First Day'].values[0,]
                last_day = hdf_ori['Last Day'].values[0,]
                burn_end_day = np.minimum(int(month_dict[month+1], 10), last_day+1)
                burn_keep_day = burn_end_day - Burn_day
                has_burn_index = Burn_day>0
                Burn_day_count = np.zeros(Burn_day.shape)
                Burn_day_count[has_burn_index] = burn_keep_day[has_burn_index]
                Burn_area = Burn_day_count*0.5*0.5 # day*km2
                # downscaled to 10 km resolution
                proj = hdf_ori.rio.crs
                x = hdf_ori['x'].values
                y = hdf_ori['y'].values
                grid_x, grid_y = np.meshgrid(x, y)
                downscale_factor = 20
                new_shape = (Burn_area.shape[0] // downscale_factor, Burn_area.shape[1] // downscale_factor)
                downscaled_array = Burn_area[:new_shape[0] * downscale_factor, :new_shape[1] * downscale_factor].reshape(new_shape + (downscale_factor, downscale_factor))
                downscaled_Burn_area = downscaled_array.max(axis=(2, 3))
                downscaled_array = grid_x[:new_shape[0] * downscale_factor, :new_shape[1] * downscale_factor].reshape(new_shape + (downscale_factor, downscale_factor))
                downscaled_grid_x = downscaled_array.mean(axis=(2, 3))
                downscaled_array = grid_y[:new_shape[0] * downscale_factor, :new_shape[1] * downscale_factor].reshape(new_shape + (downscale_factor, downscale_factor))
                downscaled_grid_y = downscaled_array.mean(axis=(2, 3))
                x = downscaled_grid_x.flatten()
                y = downscaled_grid_y.flatten()
                transformer = pyproj.Transformer.from_crs(proj, "EPSG:4326", always_xy=True)
                x_, y_ = transformer.transform(x, y)
                burn_area = downscaled_Burn_area.flatten()
                df_single_file = pd.DataFrame({'lon': x_, 'lat': y_, 'burn_area': burn_area})
                df_burn.append(df_single_file)
            df_burn = pd.concat(df_burn, axis=0, ignore_index=True)
            known_points = np.array([list(df_burn['lon']), list(df_burn['lat']), list(df_burn['burn_area'])]).T
            df_burn = griddata((known_points[:,0], known_points[:,1]), known_points[:,2], (df_grid['lon'], df_grid['lat']), method='linear')
            df_burn[np.isnan(df_burn)] = 0
            df_month = pd.DataFrame({'month': month, 'burn': df_burn})
            df_burn_all.append(df_month)
        df_burn_all = pd.concat(df_burn_all, axis=0, ignore_index=True)
        return df_burn_all