import os
import numpy as np
import pandas as pd
import xarray as xr
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import geopandas as gpd
import xgboost as xgb
from download_variable import download_LPDAAC
from define_grid import Grid_define
from preprocess_variable import Preprocess_data
from build_model import XGBoost_model
from drawing import Spatial_drawing, model_performance
import imageio

grid_obj = Grid_define(res=0.1, start_day='2023-04-01', 
                       end_day='2023-09-30', 
                       tmp_dir="/WORK/genggn_work/hechangpei/PM2.5/", 
                       shapefile="/WORK/genggn_work/hechangpei/PM2.5/China_and_World_Map_shapefiles/World/polygon/World_polygon.shp", 
                       spatial_extent=[-145, 10, -50, 70])
process_obj = Preprocess_data(grid_obj)
draw_obj = Spatial_drawing(0.1,
                          [-145, 10, -50, 70],
                          "/WORK/genggn_work/hechangpei/PM2.5/China_and_World_Map_shapefiles/World/polygon/World_polygon.shp")
                            
# grid_obj.model_grid.describe()
# 1. download varibles
# download_path = '/WORK/genggn_work/hechangpei/PM2.5/AOD/'
# while True:
#     try:
#         download_LPDAAC("MOTA", 'MCD19A2CMG.061', download_path, pm25_obj.start_day, pm25_obj.end_day,  "hcp123", 'Tejian08')
#     except Exception as e:
#         print('download error')

## 2. retrieve gridded data
# pm25 = process_obj.process_pm25("/WORK/genggn_work/hechangpei/PM2.5/site_pm2.5/OpenAQ_N18_2023_CA.csv") #
# pm25.to_csv("/WORK/genggn_work/hechangpei/PM2.5/process_result/pm25.csv", index=False)
# pop = process_obj.process_pop("/WORK/genggn_work/hechangpei/PM2.5/pop/gpw_v4_population_count_rev11_2020_2pt5_min.tif") #
# pop.to_csv("/WORK/genggn_work/hechangpei/PM2.5/process_result/pop.csv", index=False)
# burn = process_obj.process_burn('/WORK/genggn_work/hechangpei/PM2.5/Burn/MOTA/') #
# burn.to_csv("/WORK/genggn_work/hechangpei/PM2.5/process_result/burn.csv", index=False)
# aod = process_obj.process_aod('/WORK/genggn_work/hechangpei/PM2.5/AOD/MCD19A2CMG.061/') #
# aod.to_csv("/WORK/genggn_work/hechangpei/PM2.5/process_result/aod.csv", index=False)
# emission = process_obj.process_emission('/WORK/genggn_work/hechangpei/PM2.5/CEDS/CEDS_gridded_data_2021-04-21/data/') #
# emission.to_csv("/WORK/genggn_work/hechangpei/PM2.5/process_result/emission.csv", index=False)
# era5 = process_obj.process_era5('/WORK/genggn_work/hechangpei/PM2.5/ERA5/') #
# era5.to_csv("/WORK/genggn_work/hechangpei/PM2.5/process_result/era5.csv", index=False)

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

## 3. draw spatial distribution of variables
draw_obj.draw_multiple_variable({'aod':'Aerosol Optical Depth', 'burn':'Burn Area', 'pop': 'Population', 'SO2': r'SO$_{2}$', 
                                'NOx': r'NO$_{x}$', 'NH3': r'NH$_{3}$', 'OC': 'Organic Carbon', 'BC': 'Black Carbon', 
                                'u10': '10m u-component of wind', 'v10': '10m v-component of wind', 
                                'd2m': '2m dewpoint temperature', 't2m': '2m temperature', 'sp': 'Surface pressure'},
                                "/WORK/genggn_work/hechangpei/PM2.5/process_result", 
                                "/WORK/genggn_work/hechangpei/PM2.5/variable_distribution.png")

## 4. make training dataset
# pm25 = pd.read_csv("/WORK/genggn_work/hechangpei/PM2.5/process_result/pm25.csv") 
# aod = pd.read_csv("/WORK/genggn_work/hechangpei/PM2.5/process_result/aod.csv") 
# burn = pd.read_csv("/WORK/genggn_work/hechangpei/PM2.5/process_result/burn.csv") 
# emission = pd.read_csv("/WORK/genggn_work/hechangpei/PM2.5/process_result/emission.csv") 
# era5 = pd.read_csv("/WORK/genggn_work/hechangpei/PM2.5/process_result/era5.csv") 
# pop = pd.read_csv("/WORK/genggn_work/hechangpei/PM2.5/process_result/pop.csv") 

# pm25.columns = ['row', 'col', 'date', 'PM25']
# pm25['date'] = pd.to_datetime(pm25['date'])
# pm25.loc[:, 'month'] = pm25.loc[:, 'date'].dt.month
# aod['date'] = pd.to_datetime(aod['date'])
# era5['date'] = pd.to_datetime(era5['date'])

# df = pd.merge(pm25, aod, on=['row', 'col', 'date'])
# df = pd.merge(df, burn, on=['row', 'col', 'month'])
# df = pd.merge(df, emission, on=['row', 'col', 'month'])
# df = pd.merge(df, pop, on=['row', 'col'])
# df = pd.merge(df, era5, on=['row', 'col', 'date'])

# df.to_csv('/WORK/genggn_work/hechangpei/PM2.5/training_dataset.csv', index=False)

# 5. make prediction dataset
# prediction_dataset_dir = '/WORK/genggn_work/hechangpei/PM2.5/prediction_dataset/'
# if not os.path.exists(prediction_dataset_dir):
#     os.mkdir(prediction_dataset_dir)

# date_list = list(np.unique(aod['date']))
# for date in date_list:
#     df = aod[aod['date']==date]
#     df.loc[:, 'month'] = df['date'].dt.month 
#     df = pd.merge(df, burn, on=['row', 'col', 'month'])
#     df = pd.merge(df, emission, on=['row', 'col', 'month'])
#     df = pd.merge(df, pop, on=['row', 'col'])
#     df = pd.merge(df, era5, on=['row', 'col', 'date'])
#     df.to_csv(os.path.join(prediction_dataset_dir, f"{np.datetime_as_string(date, unit='D')}.csv"), index=False)
    
# 6. build model and validation
params = {'colsample_bytree': 2/3,
            'eta': 0.01,
            'eval_metric': "rmse",
            # gpu_id = 1,
            'max_depth': 8,
            # n_gpus = 1,
            'nthread': 12,
            'objective': "reg:linear",
            'subsample': 0.7,
            # 'tree_method': "gpu_hist",
            }

## build model and validation
pm25_model = XGBoost_model(params=params, 
                            independent_v=['doy', 'month', 'burn', 'aod','pop', 
                                            'BC', 'NOx', 'OC', 'NH3', 'SO2', 
                                            'u10', 'v10', 'd2m', 't2m', 'sp'], 
                            dependent_v='PM25',
                            training_data_path="/WORK/genggn_work/hechangpei/PM2.5/training_dataset.csv")
bst = pm25_model.train()
ypred = bst.predict(pm25_model.dvalid)
yobs = pm25_model.dvalid.get_label()
model_performance(yobs, ypred, '/WORK/genggn_work/hechangpei/PM2.5/validation_result.png')

## ultimate model
bst_ultimate = pm25_model.ultimate_train()
bst_ultimate.save_model(os.path.join('/WORK/genggn_work/hechangpei/PM2.5/pm25.model'))

## variable importance
# xgb.plot_importance(bst_ultimate)
# plt.show()
    
## import trained model
bst = xgb.Booster({'nthread': 12})  # init model
bst.load_model(os.path.join('/WORK/genggn_work/hechangpei/PM2.5/pm25.model'))  # load data

## 7. predict
input_dir = '/WORK/genggn_work/hechangpei/PM2.5/prediction_dataset'
output_dir = '/WORK/genggn_work/hechangpei/PM2.5/predict_result'
figure_dir = '/WORK/genggn_work/hechangpei/PM2.5/daily_map'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
if not os.path.exists(figure_dir):
    os.makedirs(figure_dir)
    
pm25_model.predict(input_dir=input_dir, output_dir=output_dir)
for filename in os.listdir(output_dir):
    draw_obj.draw_daily_map(file_path=os.path.join(output_dir, filename), 
                            variable='PM25',
                            vmin=0,
                            vmax=120,
                            title=filename.replace('.csv', ''),
                            figure_path=os.path.join(figure_dir, filename.replace('.csv', '.png')))

## make daily video
image_folder = '/WORK/genggn_work/hechangpei/PM2.5/daily_map'
video_name = '/WORK/genggn_work/hechangpei/PM2.5/daily_map_video.mp4'
images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
images.sort()
image_paths = [os.path.join(image_folder, img) for img in images]
video_writer = imageio.get_writer(video_name, fps=1/0.3)
for image_path in image_paths:
    img = imageio.imread(image_path)
    video_writer.append_data(img)
video_writer.close()




