import geopandas as gpd
import os
import pandas as pd
import rioxarray as rxr
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import seaborn as sns
import rasterio
from rasterio.transform import from_origin
import math
import pyproj
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from rasterio.transform import from_origin
from matplotlib.colors import Normalize
import matplotlib.dates as mdates
from netCDF4 import Dataset
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
file_dir = '/WORK/genggn_work/hechangpei/IAS/Burn/MOTA'
file_list = sorted([file for file in os.listdir(file_dir) if ('hdf' in file) and ('xml' not in file)])
lon = [round(x, 2) for x in [-179.95 + i * 0.1 for i in range(int((179.95 - (-179.95)) / 0.1) + 2)]]
lat = [round(x, 2) for x in [-89.95 + i * 0.1 for i in range(int((89.95- (-89.95)) / 0.1) + 1)]]
grid_lon, grid_lat = np.meshgrid(lon, lat)
month_index = ['001', '032', '060', '091', '121', '152', '182', '213', '244', '274', '305', '335']
month = list(range(1, 13))
month_dict = dict(zip(month, month_index))
map = gpd.read_file("/WORK/genggn_work/hechangpei/PM2.5/politicalboundaries_shapefile/boundaries_p_2021_v3.shp")
map = map.to_crs('EPSG:4326')

for month in list(range(1, 10)):
    # month = 1
    file_month_list = sorted([file for file in file_list if (str(month_dict[month])==file[13:16])])
    df_burn = []
    for i in range(len(file_month_list)):
        # i=200
        print(i)
        file_example = file_month_list[i]
        hdf_ori = rxr.open_rasterio(os.path.join(file_dir, file_example))
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
        df = pd.DataFrame({'lon': x_, 'lat': y_, 'burn_area': burn_area})
        df_burn.append(df)
    df_burn = pd.concat(df_burn, axis=0, ignore_index=True)

    # resample to predifined 0.1x0.1 grid
    known_points = np.array([list(df_burn['lon']), list(df_burn['lat']), list(df_burn['burn_area'])]).T
    interpolated_values = griddata((known_points[:,0], known_points[:,1]), known_points[:,2], (grid_lon, np.flip(grid_lat, axis=0)), method='linear')
    interpolated_values[np.isnan(interpolated_values)] = 0

    # draw distribution map
    vmin = np.nanmin(interpolated_values)
    vmax = np.nanmax(interpolated_values)
    norm = Normalize(vmin = vmin, vmax = vmax)
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(interpolated_values, extent = (-180, 180, -90, 90), norm = norm, cmap = 'jet')
    map.plot(ax=ax, facecolor='none', edgecolor='black')
    plt.xlim(-180, -50)
    plt.ylim(10, 85)
    ax.set_title(str(month)+'月')
    cb = fig.colorbar(im, shrink=0.8, pad=0.15)
    cb.ax.set_title('过火面积 (day·km2)')
    cb.ax.tick_params(labelsize=12)
    # plt.show()
    plt.savefig(f'/WORK/genggn_work/hechangpei/PM2.5/Burn/Burn_count_{month}.png', dpi = 300)


    
    