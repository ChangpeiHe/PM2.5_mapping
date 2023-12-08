import os
import numpy as np
import pandas as pd
import xarray as xr
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import geopandas as gpd
from download_variable import download_LPDAAC
from define_grid import Grid_define
from preprocess_variable import Preprocess_data

grid_obj = Grid_define(res=0.1, start_day='2023-04-01', 
                       end_day='2023-04-03', 
                       tmp_dir="/WORK/genggn_work/hechangpei/PM2.5/", 
                       shapefile="/WORK/genggn_work/hechangpei/PM2.5/China_and_World_Map_shapefiles/World/polygon/World_polygon.shp", 
                       spatial_extent=[-145, 10, -50, 70])
process_obj = Preprocess_data(grid_obj)
# grid_obj.model_grid.describe()
# 1. download varibles
# download_path = '/WORK/genggn_work/hechangpei/PM2.5/AOD/'
# while True:
#     try:
#         download_LPDAAC("MOTA", 'MCD19A2CMG.061', download_path, pm25_obj.start_day, pm25_obj.end_day,  "hcp123", 'Tejian08')
#     except Exception as e:
#         print('download error')

## 2. retrieve gridded data
pm25 = process_obj.process_pm25("/WORK/genggn_work/hechangpei/PM2.5/site_pm2.5/OpenAQ_N18_2023_CA.csv") #
pm25.to_csv("/WORK/genggn_work/hechangpei/PM2.5/process_result/pm25.csv", index=False)
pop = process_obj.process_pop("/WORK/genggn_work/hechangpei/PM2.5/pop/gpw_v4_population_count_rev11_2020_2pt5_min.tif") #
pop.to_csv("/WORK/genggn_work/hechangpei/PM2.5/process_result/pop.csv", index=False)
burn = process_obj.process_burn('/WORK/genggn_work/hechangpei/PM2.5/Burn/MOTA/') #
burn.to_csv("/WORK/genggn_work/hechangpei/PM2.5/process_result/burn.csv", index=False)
aod = process_obj.process_aod('/WORK/genggn_work/hechangpei/PM2.5/AOD/MCD19A2CMG.061/') #
aod.to_csv("/WORK/genggn_work/hechangpei/PM2.5/process_result/aod.csv", index=False)
emission = process_obj.process_emission('/WORK/genggn_work/hechangpei/PM2.5/CEDS/CEDS_gridded_data_2021-04-21/data/') #
emission.to_csv("/WORK/genggn_work/hechangpei/PM2.5/process_result/emission.csv", index=False)
era5 = process_obj.process_era5('/WORK/genggn_work/hechangpei/PM2.5/ERA5/') #
era5.to_csv("/WORK/genggn_work/hechangpei/PM2.5/process_result/era5.csv", index=False)

# print(pm25)
# print(pop)
# print(burn)
# print(aod)
# print(emission)
# print(era5)

# np.unique(pm25['PM25'])
# np.unique(pop['pop'])
# np.unique(burn['burn'])
# np.unique(aod['aod'])
# np.unique(emission)
# np.unique(era5)













