import os
import numpy as np
# from download_variable import download_LPDAAC
from define_grid import Grid_define
from preprocess_variable import Preprocess_data
import pandas as pd
import xarray as xr
from scipy.interpolate import griddata
import matplotlib.pyplot as plt

grid_obj = Grid_define(res=0.1, start_day='2023-04-01', end_day='2023-04-03', tmp_dir="/WORK/genggn_work/hechangpei/PM2.5/", shapefile="/WORK/genggn_work/hechangpei/PM2.5/politicalboundaries_shapefile/boundaries_p_2021_v3.shp")
process_obj = Preprocess_data(grid_obj)

# 1. download varibles
# download_path = '/WORK/genggn_work/hechangpei/PM2.5/AOD/'
# while True:
#     try:
#         download_LPDAAC("MOTA", 'MCD19A2CMG.061', download_path, pm25_obj.start_day, pm25_obj.end_day,  "hcp123", 'Tejian08')
#     except Exception as e:
#         print('download error')

## 2. retrieve gridded data
pm25 = process_obj.process_pm25("/WORK/genggn_work/hechangpei/PM2.5/site_pm2.5/OpenAQ_N18_2023_CA.csv")
pop = process_obj.process_pop("/WORK/genggn_work/hechangpei/PM2.5/pop/gpw_v4_population_count_rev11_2020_2pt5_min.tif")
burn = process_obj.process_burn('/WORK/genggn_work/hechangpei/PM2.5/Burn/MOTA/')
aod = process_obj.process_aod('/WORK/genggn_work/hechangpei/PM2.5/AOD/MCD19A2CMG.061/')
emission = process_obj.process_emission('/WORK/genggn_work/hechangpei/PM2.5/CEDS/CEDS_gridded_data_2021-04-21/data/')
era5 = process_obj.process_era5('/WORK/genggn_work/hechangpei/PM2.5/ERA5/')
print(pm25)
print(pop)
print(burn)
print(aod)
print(emission)


# df_grid = grid_obj.model_grid
# grid_obj.map_shapefile.total_bounds
# np.min(df_grid['lat'])
# data_dir = "/WORK/genggn_work/hechangpei/PM2.5/ERA5/cache-compute-0001/cache/data5/adaptor.mars.internal-1701320405.9485595-25499-7-0e0c432b-20a8-4194-8866-dbfaf096f5fb.nc"

# df_rxr = xr.open_dataset(data_dir).sel(latitude=slice(process_obj.grid_obj.lat_north+process_obj.grid_obj.res*5, process_obj.grid_obj.lat_south-process_obj.grid_obj.res*5), 
#                                        longitude=slice(process_obj.grid_obj.lon_west-process_obj.grid_obj.res*5, process_obj.grid_obj.lon_east+process_obj.grid_obj.res*5))
# x = df_rxr['longitude'].values
# y = df_rxr['latitude'].values
# grid_x, grid_y = np.meshgrid(x, y)
# time = df_rxr['time'].values.astype('datetime64[D]')
# date_list = np.unique(time)
# date_list = date_list[1:2]
# mete_variable = ['u10', 'v10', 'd2m', 't2m', 'sp']
# df_month = []
# for var in mete_variable:
#     data = df_rxr[var].values
#     df_var = []
#     for date in date_list:
#         data_day = data[time==date, :, :]
#         data_day = np.mean(data_day, axis=0)
#         known_points = np.array([list(grid_x[np.logical_not(np.isnan(data_day))]), 
#                                 list(grid_y[np.logical_not(np.isnan(data_day))]), 
#                                 list(data_day[np.logical_not(np.isnan(data_day))])]).T   
#         data_in = griddata((known_points[:,0], known_points[:,1]), known_points[:,2], 
#                                     (df_grid['lon'], df_grid['lat']), method='linear')
#         df_day = pd.DataFrame({'row':df_grid['row'], 'col':df_grid['col'], 'date': date, var: data_in})
#         df_var.append(df_day) 
#     df_var = pd.concat(df_var, axis=0, ignore_index=True)
#     df_month.append(df_var)
# df_month = pd.concat(df_month, axis=1)
# df_month = df_month.loc[:, ~df_month.columns.duplicated()]






