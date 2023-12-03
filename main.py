from download_variable import download_LPDAAC
from define_grid import Grid_define
from preprocess_variable import Preprocess_data


grid_obj = Grid_define(res=0.1, start_day='2023-04-01', end_day='2023-09-30', tmp_dir="C:/Users/hechangpei/Desktop/", shapefile="C:/Users/hechangpei/Desktop/US_CA/USA_CA.shp")
process_obj = Preprocess_data(grid_obj)
# # 1. download varibles
# download_path = '/WORK/genggn_work/hechangpei/PM2.5/AOD/'
# while True:
#     try:
#         download_LPDAAC("MOTA", 'MCD19A2CMG.061', download_path, pm25_obj.start_day, pm25_obj.end_day,  "hcp123", 'Tejian08')
#     except Exception as e:
#         print('download error')

## 2. retrieve gridded data
pm25 = process_obj.process_pm25(r"F:\清华大学地学系博士学习\课程学习\大气遥感\OpenAQ_N18_2023_CA.csv")
pop = process_obj.process_pop(r"C:\Users\hechangpei\Desktop\gpw_v4_population_count_rev11_2020_2pt5_min.tif")
print(pm25)
print(pop)
