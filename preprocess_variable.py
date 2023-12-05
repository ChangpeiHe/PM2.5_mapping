import os
import numpy as np
import pandas as pd
import rasterio
import rioxarray as rxr
import geopandas as gpd
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import pyproj
import xarray as xr


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
        df_all = df_grid[['row', 'col']]
        df_rxr = rxr.open_rasterio(data_dir).sel(y=slice(self.grid_obj.lat_north+self.grid_obj.res*5, 
                                                         self.grid_obj.lat_south-self.grid_obj.res*5), 
                                                 x=slice(self.grid_obj.lon_west-self.grid_obj.res*5, 
                                                         self.grid_obj.lon_east+self.grid_obj.res*5))
        df = df_rxr.values[0,]
        df[df == df_rxr._FillValue] = np.nan
        x = df_rxr['x'].values
        y = df_rxr['y'].values
        grid_x, grid_y = np.meshgrid(x, y)
        known_points = np.array([list(grid_x[np.logical_not(np.isnan(df))]), list(grid_y[np.logical_not(np.isnan(df))]), list(df[np.logical_not(np.isnan(df))])]).T
        df = griddata((known_points[:,0], known_points[:,1]), known_points[:,2], (df_grid['lon'], df_grid['lat']), method='linear')
        df[np.isnan(df)] = 0
        df_all.loc['pop'] = df
        return df_all
    
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
            df_all = df_grid[['row', 'col']]
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
            df_all.loc['burn'] = df_burn
            df_all.loc['month'] = month
            df_burn_all.append(df_all)
        df_burn_all = pd.concat(df_burn_all, axis=0, ignore_index=True)
        return df_burn_all
    
    def process_aod(self, data_dir):
        file_list = np.array(sorted([file for file in os.listdir(data_dir) if file.endswith(".hdf")]))
        date_list = np.array([pd.to_datetime(file[12:19], format='%Y%j').date() for file in file_list])
        day_start = pd.to_datetime(self.grid_obj.start_day, format='%Y-%m-%d').date()
        day_end = pd.to_datetime(self.grid_obj.end_day, format='%Y-%m-%d').date()
        file_list = file_list[(date_list>=day_start) & (date_list<=day_end)]
        date_list = np.array([pd.to_datetime(file[12:19], format='%Y%j').date() for file in file_list])
        df_grid = self.grid_obj.model_grid
        df_all = []
        for i, filename in enumerate(file_list):
            file_dir = os.path.join(data_dir, filename)
            hdf_ori = rxr.open_rasterio(file_dir).sel(y=slice(self.grid_obj.lat_north+self.grid_obj.res, 
                                                             self.grid_obj.lat_south-self.grid_obj.res), 
                                                      x=slice(self.grid_obj.lon_west-self.grid_obj.res, 
                                                             self.grid_obj.lon_east+self.grid_obj.res))
            AOD = hdf_ori['AOD_055'].values[0,]*0.001
            AOD[AOD == -28.672] = np.nan
            x = hdf_ori['x'].values
            y = hdf_ori['y'].values
            grid_x, grid_y = np.meshgrid(x, y)
            known_points = np.array([list(grid_x[np.logical_not(np.isnan(AOD))]), 
                                     list(grid_y[np.logical_not(np.isnan(AOD))]), 
                                     list(AOD[np.logical_not(np.isnan(AOD))])]).T    
             
            aod = griddata((known_points[:,0], known_points[:,1]), 
                                        known_points[:,2], 
                                        (df_grid['lon'], df_grid['lat']), 
                                        method='linear')
            NA_num = np.isnan(aod).sum()
            if NA_num > len(aod)*0.3:
                continue
            else:
                df = pd.DataFrame({'row':df_grid['row'], 'col':df_grid['col'], 'date': date_list[i], 'aod':aod})
                df_all.append(df)
        df_all = pd.concat(df_all, axis=0, ignore_index=True)
        return df_all

    def process_emission(self, data_dir):
        df_grid = self.grid_obj.model_grid
        species = ['BC', 'NOx', 'OC', 'NH3', 'SO2']
        df_all = []
        for i in range(len(species)):
            # i=0
            file_dir = os.path.join(data_dir, species[i], 'individual_files')
            filename = np.array(sorted([file for file in os.listdir(file_dir) if 'em-anthro' in file and '200001-201912' in file]))[0]
            df_rxr = xr.open_dataset(os.path.join(file_dir, filename)).sel(lat=slice(self.grid_obj.lat_south-self.grid_obj.res*5, self.grid_obj.lat_north+self.grid_obj.res*5), 
                                                                           lon=slice(self.grid_obj.lon_west-self.grid_obj.res*5, self.grid_obj.lon_east+self.grid_obj.res*5))
            x = df_rxr['lon'].values
            y = df_rxr['lat'].values
            grid_x, grid_y = np.meshgrid(x, y)
            df = df_rxr[f'{species[i]}_em_anthro'].values
            df = np.sum(df, axis=1)
            df = df[-13:-1, :, :]
            df_species = []
            for j in range(12):
                month = j+1
                df_month = df[j, :, :]
                known_points = np.array([grid_x.flatten(), 
                                        grid_y.flatten(), 
                                        df_month.flatten()]).T
                emission = griddata((known_points[:,0], known_points[:,1]), known_points[:,2], 
                                            (df_grid['lon'], df_grid['lat']), method='linear')
                df_month = pd.DataFrame({'row':df_grid['row'], 'col':df_grid['col'], 'month': month, species[i]: emission})
                df_species.append(df_month)
            df_species = pd.concat(df_species, axis=0, ignore_index=True)
            df_all.append(df_species)
        df_all = pd.concat(df_all, axis=1)
        df_all = df_all.loc[:, ~df_all.columns.duplicated()]
            # df_all.loc[species[i]] = emission
        return(df_all)

    def process_era5(self, data_dir):
        df_grid = self.grid_obj.model_grid
        file_list = os.listdir(data_dir)
        df_all = []
        for filename in file_list:
            df_rxr = xr.open_dataset(os.path.join(data_dir, filename)).sel(latitude=slice(self.grid_obj.lat_north+self.grid_obj.es*5, self.grid_obj.lat_south-self.grid_obj.res*5), 
                                                                       longitude=slice(self.grid_obj.lon_west-self.grid_obj.res*5, self.grid_obj.lon_east+self.grid_obj.res*5))
            x = df_rxr['longitude'].values
            y = df_rxr['latitude'].values
            grid_x, grid_y = np.meshgrid(x, y)
            time = df_rxr['time'].values.astype('datetime64[D]')
            date_list = np.unique(time)
            mete_variable = ['u10', 'v10', 'd2m', 't2m', 'sp']
            df_month = []
            for var in mete_variable:
                data = df_rxr[var].values
                df_var = []
                for date in date_list:
                    data_day = data[time==date, :, :]
                    data_day = np.mean(data_day, axis=0)
                    known_points = np.array([list(grid_x[np.logical_not(np.isnan(data_day))]), 
                                            list(grid_y[np.logical_not(np.isnan(data_day))]), 
                                            list(data_day[np.logical_not(np.isnan(data_day))])]).T   
                    data_in = griddata((known_points[:,0], known_points[:,1]), known_points[:,2], 
                                                (df_grid['lon'], df_grid['lat']), method='linear')
                    df_day = pd.DataFrame({'row': df_grid['row'], 'col': df_grid['col'], 'date': date, var: data_in})
                    df_var.append(df_day) 
                df_var = pd.concat(df_var, axis=0, ignore_index=True)
                df_month.append(df_var)
            df_month = pd.concat(df_month, axis=1)
            df_month = df_month.loc[:, ~df_month.columns.duplicated()]
            df_all.append(df_month)
        df_all = pd.concat(df_all, axis=0, ignore_index=True)
        return df_all


        