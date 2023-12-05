import os
import numpy as np
# from download_variable import download_LPDAAC
from define_grid import Grid_define
from preprocess_variable import Preprocess_data
import pandas as pd


grid_obj = Grid_define(res=0.1, start_day='2023-04-01', end_day='2023-04-03', tmp_dir="/WORK/genggn_work/hechangpei/PM2.5/", shapefile="/WORK/genggn_work/hechangpei/PM2.5/US_CA/USA_CA.shp")
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
print(pm25)
print(pop)
print(burn)
print(aod)
print(emission)




