from pymodis.downmodis import downModis
from earthdata_download.api import EarthdataAPI
import os
import datetime
import re
import warnings
warnings.filterwarnings("ignore")


def download_LPDAAC(path, product, download_path, start_day, end_day, user, password, tiles=None):
    '''
        download datasets from lp_daac repository
        path: refer to https://e4ftl01.cr.usgs.gov/, e.g., MOLA
        product: refer to https://e4ftl01.cr.usgs.gov/, e.g., MCD19A2.061
        download datasets from LP_DAAC repository
        start_day, e.g., "2023-09-18"
        end_day, e.g., "2023-10-19"
        user
        password
        tiles: e.g., 'h26v04', 'h27v04'
        download_path
    '''
    
    download_path = os.path.join(download_path, product)
    if not os.path.exists(download_path):
        os.makedirs(download_path)
    down_modis = downModis(
        destinationFolder=download_path,
        user=user,
        password=password,
        path=path,
        product=product,
        today=start_day,
        enddate=end_day,
        tiles=tiles)
    down_modis.connect()
    down_modis.downloadsAllDay()



def download_earthdata(product, download_path, start_day, end_day, extent, user, password, filetype):
    '''
        download datasets from Earthdata repository
        product: e.g., 'HLSL30'
        start_day, e.g., "2023-09-18"
        end_day, e.g., "2023-10-19"
        extent: spatial extent e.g., extent={'xmin': 160, 
                                            'xmax': 170, 
                                            'ymin': 10,
                                            'ymax': 40
                                            }
        user
        password
        download_path
        filetype: e.g., '.tif' '.nc' '.h5'
    '''

    download_path = os.path.join(download_path, product)
    if not os.path.exists(download_path):
        os.makedirs(download_path)
    myapi = EarthdataAPI(user, password)
    results = myapi.query(
        short_name=product, 
        start_date=datetime.datetime.strptime(start_day, '%Y-%m-%d'),
        end_date=datetime.datetime.strptime(end_day, "%Y-%m-%d"), 
        extent=extent
    )
    url_list = []
    for result in results:
        pattern = rf'http[^\s]*{filetype}'
        matches = re.findall(pattern, str(result))
        url_list.extend(matches)       
    print(url_list)
    myapi.download(url_list, download_dir=download_path, skip_existing=True)


if __name__ == "__main__":
    # download_LPDAAC("MOTA", 'MCD19A2.061', download_path, start_day, end_day, tiles, "hcp123", 'Tejian08')
    download_earthdata('OMNO2', '/WORK/genggn_work/hechangpei/PM2.5/NO2/', '2023-09-01', '2023-09-02', 
                       {'xmin': 160, 'xmax': 170, 'ymin': 10, 'ymax': 40}, 
                       "hcp123", 'Tejian08', '.he5')

