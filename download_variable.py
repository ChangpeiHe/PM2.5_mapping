from pymodis.downmodis import downModis
from earthdata_download.api import EarthdataAPI
import os

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

def download_LPDAAC(path, product, download_path, start_day, end_day, user, password, tiles=None):
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


'''
download datasets from Earthdata repository
product: e.g., 'HLSL30'
start_day, e.g., "2023-09-18"
end_day, e.g., "2023-10-19"
user
password
download_path
filetype: e.g., '.tif' '.nc'
'''
def download_earthdata(product, download_path, start_day, end_day, user, password, filetype):
    download_path = os.path.join(download_path, product)
    if not os.path.exists(download_path):
        os.makedirs(download_path)
    myapi = EarthdataAPI(user, password)
    results = myapi.query(
        short_name=product, 
        start_date=datetime.datetime.strptime(start_day, '%Y-%m-%d'),
        end_date=datetime.datetime.strptime(end_day, "%Y-%m-%d"), 
        extent={
            'xmin': df_basic_info['xmin'][i],
            'xmax': df_basic_info['xmax'][i],
            'ymin': df_basic_info['ymin'][i],
            'ymax': df_basic_info['ymax'][i]
        }
    )
    print(f"Totally {len(results)} files")
    url_list = []
    for result in results:
        pattern = rf'http[^\s]*{file_type}'
        matches = re.findall(pattern, str(result))
        url_list.extend(matches)       
    myapi.download(url_list, download_dir=download_dir, skip_existing=True)

if __name__ == "__main__":
    download_LPDAAC("MOTA", 'MCD19A2.061', download_path, start_day, end_day, tiles, "hcp123", 'Tejian08')
    download_LPDAAC("MOTA", 'MCD19A2.061', download_path, start_day, end_day, tiles, "hcp123", 'Tejian08')

