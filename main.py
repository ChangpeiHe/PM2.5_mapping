from download_variable import *

start_day = '2023-04-01'
end_day = '2023-09-30'
download_path = '/WORK/genggn_work/hechangpei/PM2.5/AOD/'

while True:
    try:
        download_LPDAAC("MOTA", 'MCD19A2CMG', download_path, start_day, end_day,  "hcp123", 'Tejian08')
    except Exception as e:
        print('download error')
