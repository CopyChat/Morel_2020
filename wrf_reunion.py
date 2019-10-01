"""
to validation and analysis WRF over la Reunion.
"""

__version__ = f'Version 1.0  \nTime-stamp: <2019-09-24>'
__author__ = "ChaoTANG@univ-reunion.fr"

import sys

DIR = f'/Users/ctang/Microsoft_OneDrive/OneDrive/CODE/Morel_2020'
sys.path.append(DIR)
import GEO_PLOT

# ----------------------DataBase---------------------------
pr_MeteoFrance = f'SELECT pr.station_id, dt as DateTime, pr.pr, longitude, latitude, altitude ' \
                 f'FROM SWIO.precip_1day_meteoFrance as pr ' \
                 f'inner JOIN station_meteoFrance_2 as station on pr.station_id=station.station_id;'

# ----------------------------- plot station mean -----------------------------

swdown_MeteoFrance = f'SELECT rg.station_id, dt as DateTime, rg.rg as swdown, longitude, latitude, altitude ' \
                     f'FROM SWIO.rg_1hour_meteoFrance as rg ' \
                     f'inner JOIN SWIO.station_meteoFrance_2 as station on rg.station_id=station.station_id;'

# swdown_MeteoFrance_df = GEO_PLOT.query_data(mysql_query=swdown_MeteoFrance)
# swdown_station_mean = swdown_MeteoFrance_df.groupby('station_id').mean()
# GEO_PLOT.plot_station_value(lon=swdown_station_mean['longitude'], lat=swdown_station_mean['latitude'],
#                             value=swdown_station_mean['swdown'], cbar_label='swdown (W/m2)',
#                             cbar_min=40, cbar_max=100)

# ----------------------------- plot diurnal cycle -----------------------------

ssr_diurnal = f'SELECT station_id, dt as DateTime,' \
              f'ssr_meteofrance, ssr_sarah_e, ssr_wrf35 ' \
              f'FROM SWIO.wrf35_valid_hour_meteofrance ' \
              f'where dt>="2004-11-01" and dt<="2005-04-30 23:59:00";'

# diurnal_data = GEO_PLOT.query_data(ssr_diurnal)
# GEO_PLOT.plot_hourly_curve_by_month(diurnal_data.groupby(diurnal_data.index).mean(),
#                                     columns=['ssr_meteofrance', 'ssr_sarah_e', 'ssr_wrf35'])

# ----------------------------- statistics @ station -----------------------------

ssr_daytime = f'SELECT station_id, dt as DateTime,' \
              f'ssr_meteofrance, ssr_sarah_e, ssr_wrf35, longitude, latitude ' \
              f'FROM SWIO.wrf35_valid_hour_meteofrance ' \
              f'where dt>="2004-11-01" and dt<="2005-04-30 23:59:00" and ' \
              f'hour(dt)>=8 and hour(dt)<=17;'

ssr_daytime_df = GEO_PLOT.query_data(ssr_daytime)
ssr_daytime_station_mean = ssr_daytime_df.groupby('station_id').mean()

# mean error:
GEO_PLOT.plot_station_value(lon=ssr_daytime_station_mean['longitude'], lat=ssr_daytime_station_mean['latitude'],
                            value=ssr_daytime_station_mean['ssr_wrf35'] - ssr_daytime_station_mean['ssr_meteofrance'],
                            cbar_label=r'WRF35 - MétéoFrance SSR $(W/m^2)$', cbar_min=-50, cbar_max=50)
print(f'got data')

