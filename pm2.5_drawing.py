import os
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.colors import LinearSegmentedColormap
from pyproj import CRS
import pyproj
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, Bbox
import math
from matplotlib.ticker import ScalarFormatter

# class Draw_pm25:
#     """
#     draw annual mean PM25, SO4 2-, NO3-, NH4+, OM, BC.
#     species: types of pm2.5 components to be drawed, e.g., ['PM2.5', 'SO4', 'NO3', 'NH4', 'OM', 'BC'], 'list' 
#     year: which years to draw, e.g., list(range(2000, 2023)), 'list'
#     color_scheme: e.g., plasma, viridis, inferno, magma, cividis, 'str'
#     color_list: self-defined color scheme, [''], 'list'
#     draw_extent: e.g., (73, 18, 136, 54), 'tuple'
#     jiuduan_extent: e.g., (73, 18, 136, 54), 'tuple'
    
#     """
#     res = 12000
#     def __init__(self, map_path, data_path, figure_dir, species, species_label, year, draw_extent, jiuduan_extent, color_scheme=None, color_list=None) -> None:
#         self.map_path = map_path
#         self.data_path = data_path
#         self.species = species
#         self.species_label = species_label
#         self.year = year
#         self.color_scheme = color_scheme
#         self.color_list = color_list
#         self.shapefile = gpd.read_file(self.map_path)
#         self.shapefile['color'] = ['#0D151D', '#94A0AA', '#94A0AA', '#0D151D', '#94A0AA', '#4496C3','#4496C3', '#4496C3','#4496C3'] 
#         self.shapefile['linetype'] = ['dashed','solid','solid','solid','solid','solid','solid','solid','solid'] 
#         self.figure_dir = figure_dir
#         self.crs = CRS.from_proj4('+proj=lcc +lat_1=25 +lat_2=40 +lon_0=110')
#         self.draw_extent = draw_extent
#         self.jiuduan_extent = jiuduan_extent
#         self.hw_ratio = (len(self.species)*(self.draw_extent[3]-self.draw_extent[1]))/(len(self.year)*(self.draw_extent[2]-self.draw_extent[0]))
#         self.jiuduan_cmap = LinearSegmentedColormap.from_list('jiuduan_colormap', [plt.cm.colors.hex2color(hex_color) for hex_color in ['#4496C3', '#4496C3','#4496C3', '#C1DA7E', '#94A0AA', '#4496C3', '#0D151D']])
#         self.shapefile = self.shapefile.iloc[::-1].reset_index(drop=True)
#         if self.shapefile.crs is None:
#             self.shapefile = self.shapefile.set_crs('EPSG:4326')
#             self.shapefile = self.shapefile.to_crs(self.crs)
#         else:
#             self.shapefile = self.shapefile.to_crs(self.crs)
#         if self.color_scheme is None:
#             self.cmap = LinearSegmentedColormap.from_list('colormap', [plt.cm.colors.hex2color(hex_color) for hex_color in self.color_list], N=len(self.color_list))
#         else:
#             self.cmap = self.color_scheme
#         df = pd.read_csv(os.path.join(self.data_path, os.listdir(self.data_path)[0]))
#         transformer = pyproj.Transformer.from_crs('EPSG:4326', self.crs, always_xy=True)    
#         df['X_Lon'], df['Y_Lat'] = transformer.transform(df['X_Lon'], df['Y_Lat'])
#         self.lon_west = np.min(df['X_Lon'])
#         self.lon_east = np.max(df['X_Lon'])
#         self.lat_north = np.max(df['Y_Lat'])
#         self.lat_south = np.min(df['Y_Lat'])
#         self.height = round((self.lat_north-self.lat_south)/self.res)+1
#         self.width = round((self.lon_east-self.lon_west)/self.res)+1
        
#     def draw(self):
#         fig, axs = plt.subplots(len(self.species), len(self.year), figsize=(len(self.year), len(self.year)*self.hw_ratio))
#         fig.subplots_adjust(left=0.15, right=0.85, top=0.85, bottom=0.15, wspace=0, hspace=0)
#         for i, species in enumerate(self.species):
#             if species == 'PM2.5':
#                 vmax = 100
#             elif species == 'BC':
#                 vmax = 10
#             else:
#                 vmax = 30
#             axs[i, 0].set_ylabel(self.species_label[i], rotation=0, ha='right', va='center', fontsize=8)
#             for j, year in enumerate(self.year):
#                 df = self.get_species_data(species=species, year=year)
#                 norm = Normalize(vmin=0, vmax=vmax+vmax*0.05, clip=True)
#                 im = axs[i, j].matshow(df, extent=(self.lon_west-self.res/2, self.lon_east+self.res/2, self.lat_south-self.res/2, self.lat_north+self.res/2), 
#                                        cmap=self.cmap, origin='upper', norm=norm)
#                 if i == 0:
#                     axs[i, j].set_title(f'{year}', fontsize=8, fontweight='bold')
#                 else:
#                     axs[i, j].set_title('')
#                 self.shapefile.plot(ax=axs[i, j], color=self.shapefile['color'], linestyle=self.shapefile['linetype'], linewidth=0.2)
#                 axs[i, j].set_xticks([])
#                 axs[i, j].set_yticks([])
#                 axs[i, j].set_xlim(self.draw_extent[0], self.draw_extent[2])
#                 axs[i, j].set_ylim(self.draw_extent[1], self.draw_extent[3])
#                 jiuduan_ratio = (self.draw_extent[2]-self.draw_extent[0])/(self.draw_extent[3]-self.draw_extent[1])
#                 cax = inset_axes(axs[i, j], width=0.11, height=jiuduan_ratio*0.11-0.01, loc='lower right', borderpad=0)
#                 cax.set_xticks([])
#                 cax.set_yticks([])
#                 cax.set_xlim(self.jiuduan_extent[0], self.jiuduan_extent[1])
#                 cax.set_ylim(self.jiuduan_extent[2], self.jiuduan_extent[3])
#                 cax.imshow(df, extent=(self.lon_west-self.res/2, self.lon_east+self.res/2, self.lat_south-self.res/2, self.lat_north+self.res/2), 
#                                     cmap=self.cmap, origin='upper', norm=norm)
#                 self.shapefile.plot(ax=cax, color=self.shapefile['color'], linestyle=self.shapefile['linetype'], linewidth=0.2)
#             if species == 'PM2.5' or species == 'BC':
#                 position = fig.add_axes([0.86, axs[i, -1].get_position().y0+0.01, 0.01, axs[i, -1].get_position().y1-axs[i, -1].get_position().y0-0.02])
#                 cb = fig.colorbar(im, cax=position)
#                 cb.ax.tick_params(labelsize=7)
#                 if species == 'PM2.5':
#                     cb.ax.set_title(r'$\mu$g/m$^{3}$', fontdict={"size":8, "color":"k"}, pad=8)
#                     cb.set_ticks(np.arange(0, 100+25, 25))
#                 else:
#                     cb.set_ticks(np.arange(0, 10+2.5, 2.5))
#             elif species == 'OM':
#                 position = fig.add_axes([0.86, axs[i, -1].get_position().y0+0.01, 0.01, axs[i-3, -1].get_position().y1-axs[i, -1].get_position().y0-0.02])
#                 cb = fig.colorbar(im, cax=position)
#                 cb.ax.tick_params(labelsize=7)
#                 cb.set_ticks(np.arange(0, 30+7.5, 7.5))
#         plt.savefig(os.path.join(self.figure_dir, 'figure.png'), dpi=1000)
        
#     def get_species_data(self, species, year):
#         df = pd.read_csv(os.path.join(self.data_path, str(year) + '_PM25_and_species.csv'))
#         transformer = pyproj.Transformer.from_crs('EPSG:4326', self.crs, always_xy=True)    
#         x, y = transformer.transform(df['X_Lon'], df['Y_Lat'])
#         data_array = np.full((self.height, self.width), fill_value=np.nan)
#         col_index = ((x-self.lon_west)//self.res).astype('int')
#         row_index = ((self.lat_north-y)//self.res).astype('int')
#         data_array[row_index, col_index] = df[species]
#         return data_array  
        

# if __name__ == "__main__":    
#     draw_full_species = Draw_pm25(r"F:\GIS\China_and_World_Map_shapefiles\China_and_World_Map_shapefiles\China\China_polyline\China_polyline.shp",
#                                     'C:/Users/hechangpei/Desktop/PM2.5_result', 
#                                     'C:/Users/hechangpei/Desktop/', 
#                                     ['PM2.5', 'SO4', 'NO3', 'NH4', 'OM', 'BC'],
#                                     [r'PM$_{2.5}$', r'SO$_{4}^{2-}$', r'NO$_{3}^{-}$', r'NH$_{4}^{+}$', 'OM', 'BC'],
#                                     list(range(2017, 2023)),
#                                     (-3400000, 2050000, 2500000, 6300000),
#                                     (-409547, 1496389, 462360, 2783500),
#                                     color_scheme=None,
#                                     color_list= ['#1F499F', '#2A50A3', '#325CA8', '#3D72B8', '#4796D2', '#42AADE', '#3FBEEA', '#68C5D5', '#7FCAC2', '#96CFAF', '#ACD697', '#C1DA7E', '#D4DF65', '#E0E54B', '#EEE32E', '#F3C91D', '#F3A81A', '#EF8517', '#E76411', '#E61118', '#E61118']
#                                 )  
#     draw_full_species.draw()


class Draw_variable:
    '''
        draw spatial distributions of predictors 
    '''
    
    crs = 'EPSG:4326'
    color_scheme = [['#1F499F', '#2A50A3', '#325CA8', '#3D72B8', '#4796D2', '#42AADE', '#3FBEEA', '#68C5D5', '#7FCAC2', '#96CFAF', '#ACD697', '#C1DA7E', '#D4DF65', '#E0E54B', '#EEE32E', '#F3C91D', '#F3A81A', '#EF8517', '#E76411', '#E61118', '#E61118'], ]
    def __init__(self, res, variables, draw_extent, map_path, data_dir, figure_dir) -> None:
        self.variables = variables
        self.res = res
        self.data_dir = data_dir
        self.map_path = map_path
        self.figure_dir = figure_dir
        self.draw_extent = draw_extent
        self.shapefile = gpd.read_file(self.map_path)
        self.file_list = os.listdir(data_dir)
        self.rows = int(math.sqrt(len(self.variables)))
        self.cols = int(math.sqrt(len(self.variables)))
        if len(self.variables) % self.rows != 0:
            self.cols += 1
        self.hw_ratio = (self.rows*(self.draw_extent[3]-self.draw_extent[1]))/(self.cols*(self.draw_extent[2]-self.draw_extent[0]))
        if self.shapefile.crs is None:
            self.shapefile = self.shapefile.set_crs('EPSG:4326')
            self.shapefile = self.shapefile.to_crs(self.crs)
        else:
            self.shapefile = self.shapefile.to_crs(self.crs)
            
    def col_to_lon(self, col):
        lon = self.res/2+(col-1)*self.res-180
        return lon

    def row_to_lat(self, row):
        lat = 90-self.res/2-(row-1)*self.res
        return lat
    
    def transfer_draw(self, vmax, vmin, type):
        if (vmax-vmin) < 1:
            base = 10 ** math.floor(-math.log10(vmax-vmin))
        else:
            base = 10 ** math.floor(math.log10(vmax-vmin))
        if type=='vmax':
            result = base * math.ceil(vmax / base)
        elif type=='vmin':
            result = base * math.floor(vmin // base)
        return result

    def draw(self):
        fig, axs = plt.subplots(self.rows, self.cols, figsize=(self.cols*2, self.cols*self.hw_ratio*2))
        print(self.cols*self.hw_ratio, self.cols)
        fig.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, wspace=0.05, hspace=0)
        gen_axs = (i for i in range(len(self.variables))) 
        
        for filename in self.file_list:
            df = pd.read_csv(os.path.join(self.data_dir, filename))
            variables = list(self.variables.keys())
            variables = list(set(df.columns) & set(variables))
            if len(variables)==0:
                continue
            lon_west = self.col_to_lon(np.min(df['col']))
            lon_east = self.col_to_lon(np.max(df['col']))
            lat_north = self.row_to_lat(np.min(df['row']))
            lat_south = self.row_to_lat(np.max(df['row']))
            height = round((lat_north-lat_south)/self.res)+1
            width = round((lon_east-lon_west)/self.res)+1
            for variable in variables:
                i = next(gen_axs)
                data = df.groupby(['row', 'col'])[variable].mean().reset_index()
                data_array = np.full((height, width), fill_value=np.nan)
                row_index = data['row']-np.min(data['row'])
                col_index = data['col']-np.min(data['col'])
                data_array[row_index, col_index] = data[variable]
                vmax = data[variable].quantile(0.99)
                vmin = data[variable].quantile(0.01)
                vmax = self.transfer_draw(vmax, vmin, 'vmax')
                vmin = self.transfer_draw(vmax, vmin, 'vmin')
                norm = Normalize(vmin=vmin, vmax=vmax+(vmax-vmin)*0.05, clip=True)
                cmap = LinearSegmentedColormap.from_list('colormap', [plt.cm.colors.hex2color(hex_color) for hex_color in self.color_scheme[0]], N=100)
                im = axs.flat[i].matshow(data_array, extent=(lon_west-self.res/2, lon_east+self.res/2, lat_south-self.res/2, lat_north+self.res/2), 
                                    cmap=cmap, origin='upper', norm=norm)
                axs.flat[i].set_title(self.variables[variable], fontsize=8, fontweight='bold', pad=3)
                self.shapefile.plot(ax=axs.flat[i], facecolor='none', edgecolor='black', linewidth=0.5)
                axs.flat[i].set_xticks([])
                axs.flat[i].set_yticks([])
                axs.flat[i].set_xlim(self.draw_extent[0], self.draw_extent[2])
                axs.flat[i].set_ylim(self.draw_extent[1], self.draw_extent[3])
                cb = plt.colorbar(im, ax=axs.flat[i], shrink=0.7, pad=0.02)
                cb.ax.tick_params(labelsize=7, pad=0.5)
                cb.set_ticks(np.arange(vmin, vmax+(vmax-vmin)*0.01, (vmax-vmin)/5))
                formatter = ScalarFormatter()
                formatter.set_scientific(True)
                formatter.set_powerlimits((-2, 3))
                cb.formatter = formatter
                cb.update_ticks()
                offset_text = cb.ax.yaxis.get_offset_text()
                offset_text.set_size(7)
                offset_text.set(va='bottom') 
        # plt.show()
        plt.savefig(os.path.join(self.figure_dir, 'figure.png'), dpi=1000)

if __name__ == "__main__":    
    draw_varaible = Draw_variable(0.1,
                                {'aod':'Aerosol Optical Depth', 'burn':'Burn Area', 'pop': 'Population', 'SO2': r'SO$_{4}^{2-}$', 
                                 'NO3': r'NO$_{3}^{-}$', 'NH4': r'NH$_{4}^{+}$', 'OC': 'Organic Carbon', 'BC': 'Black Carbon', 
                                 'PM25':r'PM$_{2.5}$', 'u10': '10m u-component of wind', 'v10': '10m v-component of wind', 
                                 'd2m': '2m dewpoint temperature', 't2m': '2m temperature', 'sp': 'Surface pressure'},
                                [-145, 10, -50, 70],
                                "F:/GIS/China_and_World_Map_shapefiles/China_and_World_Map_shapefiles/World/polygon/World_polygon.shp",
                                "C:/Users/hechangpei/Desktop/process_result", 
                                'C:/Users/hechangpei/Desktop/'             
                                )  
    draw_varaible.draw()
