# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 16:26:10 2023

@author: hechangpei
"""
from shapely.geometry import Point
import numpy as np
import pandas as pd
import os
import geopandas as gpd
# import rioxarray as rxr
# import rasterio
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
NA_shp =  gpd.read_file(r"F:\清华大学地学系博士学习\课程学习\大气遥感\politicalboundaries_shapefile\boundaries_p_2021_v3.shp")
NA_shp = NA_shp.to_crs('EPSG:4326')

df_pm =  pd.read_csv(r"F:\清华大学地学系博士学习\课程学习\大气遥感\OpenAQ_N18_2023_CA.csv")
df_pm = df_pm[(df_pm['Lon']<-50) & (df_pm['Lat']>14) & (df_pm['Lat']<84)]
np.unique(df_pm['Country'])
# BZ: 伯利兹 (Belize)
# CA: 加拿大 (Canada)
# GT: 危地马拉 (Guatemala)
# MX: 墨西哥 (Mexico)
# PR: 波多黎各 (Puerto Rico)
# US: 美国 (United States)
# 1. temporal
df_pm['Date'] = pd.to_datetime(df_pm['Year'].astype(str) + df_pm['DOY'].astype(str), format='%Y%j')
np.max(df_pm['Date']) # 20230930
np.min(df_pm['Date']) # 20230101
df_pm.drop_duplicates(subset=['Lon', 'Lat']) # 1901个站点
df_pm_mean = df_pm.groupby(['Date'])[['PM25']].mean().reset_index()

fig, ax = plt.subplots(figsize=(6, 5))
plt.plot(list(df_pm_mean['Date']), list(df_pm_mean['PM25']), color='#47558e', marker='o')
plt.xlabel('Month')
plt.ylabel('PM2.5(ug/m3)')
plt.xticks(rotation=30)
plt.savefig(os.path.join(r"F:\清华大学地学系博士学习\课程学习\大气遥感\pm_obs.png"), dpi=500)
plt.close()

# 分国家画图
df_pm_Country = df_pm.groupby(['Date', 'Country'])[['PM25']].mean().reset_index()
Country_list = list(np.unique(df_pm['Country']))
color_list = ['#B5CDE2', '#DDCCD0', '#706476', '#F7EFE9', '#E4A8A8', '#516A76']
label_list = Country_list
fig, ax = plt.subplots(figsize=(20, 10))
for i in range(len(Country_list)):
    # i=0
    df = df_pm_Country[df_pm_Country['Country'] == Country_list[i]]
    ax.plot(list(df['Date']), list(df['PM25']), label=label_list[i], marker='o', color=color_list[i])
plt.legend()
ax.grid(alpha=0.3, color='grey')
ax.set_ylabel('PM2.5 (μg/m3)', fontsize=20, loc='center')
ax.set_xlabel('日期', fontsize=20)
ax.tick_params(axis='both', labelsize=20)
ax.tick_params(axis='both', labelsize=20)
# plt.xlim(df['日期'].min()-timedelta(days=1), df['日期'].max()+timedelta(days=1))
plt.gca().xaxis.set_major_locator(plt.MaxNLocator(10))
plt.gcf().autofmt_xdate()
# plt.show()
plt.savefig(r"F:\清华大学地学系博士学习\课程学习\大气遥感\pm_obs_country.png", dpi=300)
plt.close()

## 只选择美国加拿大
df_pm_Country = df_pm.groupby(['Date', 'Country'])[['PM25']].mean().reset_index()
Country_list = ['CA', 'US']
color_list = ['#706476', '#E4A8A8']
label_list = Country_list
fig, ax = plt.subplots(figsize=(20, 10))
for i in range(len(Country_list)):
    # i=0
    df = df_pm_Country[df_pm_Country['Country'] == Country_list[i]]
    ax.plot(list(df['Date']), list(df['PM25']), label=label_list[i], marker='o', color=color_list[i])

ax.axvline(x=pd.to_datetime('2023-05-12'), color='red', linestyle='--', label='发现野火')
plt.legend()
ax.grid(alpha=0.3, color='grey')
ax.set_ylabel('PM2.5 (μg/m3)', fontsize=20, loc='center')
ax.set_xlabel('日期', fontsize=20)
ax.tick_params(axis='both', labelsize=20)
ax.tick_params(axis='both', labelsize=20)
# plt.xlim(df['日期'].min()-timedelta(days=1), df['日期'].max()+timedelta(days=1))
plt.gca().xaxis.set_major_locator(plt.MaxNLocator(10))
plt.gcf().autofmt_xdate()
# plt.show()
plt.savefig(r"F:\清华大学地学系博士学习\课程学习\大气遥感\pm_obs_us_ca.png", dpi=300)
plt.close()

## 站点空间分布情况
df_site = df_pm.drop_duplicates(subset=['Lon', 'Lat'])[['Lon', 'Lat']]
# data = pd.merge(df, df_basic_info[['longitude', 'latitude', 'Company']], on='Company')
geometry = [Point(xy) for xy in zip(df_site['Lon'], df_site['Lat'])]
gdf = gpd.GeoDataFrame(df_site, geometry=geometry, crs='EPSG:4326')
# gdf = gdf[gdf.geometry.within(jingjinji_map.geometry.unary_union)]
fig, ax = plt.subplots(figsize=(12, 12))
NA_shp.plot(ax=ax, color='lightgray', edgecolor='black')
plt.xlim(-180, -50)
plt.ylim(10, 85)
# ax.set_extent([-180, -50, 10, 85])
plt.scatter(gdf['Lon'], gdf['Lat'], alpha=0.7, color='#E4A8A8', edgecolors='k', label='地面监测站点')
plt.legend(loc='upper left', fontsize=18)
# plt.show()
plt.savefig(r"F:\清华大学地学系博士学习\课程学习\大气遥感\pm_obs_site.png", dpi=300)
plt.close()


###



