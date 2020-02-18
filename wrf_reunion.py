"""
to validation and analysis WRF over la Reunion.
"""

__version__ = f'Version 1.0  \nTime-stamp: <2019-09-24>'
__author__ = "ChaoTANG@univ-reunion.fr"

import sys
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
from labellines import labelLines

DIR = f'/Users/ctang/Microsoft_OneDrive/OneDrive/CODE/Morel_2020'
sys.path.append(DIR)
import GEO_PLOT

# =========================== definition: plot flags ===============================
# ----------------------------- functions -----------------------------


def get_statistics(gg):
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

    bias = np.mean(gg['swdown_wrf41'] - gg['ssr_meteofrance'])
    r2 = r2_score(gg['swdown_wrf41'], gg['ssr_meteofrance'])
    mae = mean_absolute_error(gg['swdown_wrf41'], gg['ssr_meteofrance'])
    rmse = np.sqrt(mean_squared_error(gg['swdown_wrf41'], gg['ssr_meteofrance']))
    corr = np.corrcoef(gg['swdown_wrf41'], gg['ssr_meteofrance'])[0, 1]

    return pd.Series(dict(MEAN_BIAS=bias, R2=r2, MAE=mae, RMSE=rmse, corr=corr))


# ----------------------------- plot station mean -----------------------------
meteofrance_missing = 0
bias = 0
wrf41_vs_wrf35 = 0
station_mean_hourly_bias = 0
mean = 0
diurnal = 0
diurnal_bias = 0
statistics = 0
bias_distribution = 0
seasonal_bias = 0
ssr_altitude = 0
hourly_bias_at_different_stations = 0
monthly_mean_bias_at_station = 0
monthly_circulation_era5 = 1
where_and_when_cloud = 0
monthly_mean_bias_clustering = 1
# ----------------------------- code options -----------------------------

if meteofrance_missing:
    ssr_MeteoFrance = f'SELECT station_id, dt as DateTime, ssr_meteofrance as ssr, longitude, latitude ' \
                      f'from SWIO.SSR_Hourly_MeteoFrance_WRF35_WRF41_SARAH_E ' \
                      f'where dt>="2004-11-01" and dt<="2005-04-30 23:59:00" and ' \
                      f'hour(dt)>=8 and hour(dt)<=17;'

    ssr_MeteoFrance_df = GEO_PLOT.query_data(mysql_query=ssr_MeteoFrance, remove_missing_data=False)
    ssr_station_mean = ssr_MeteoFrance_df.groupby('station_id').mean()

    # ----------------------------- missing timestep -----------------------------
    # to plot a map of all these meteofrance with only missing value

    GEO_PLOT.station_data_missing_map_hourly_by_month(df=ssr_MeteoFrance_df,
                                                      station_id='station_id', columns='ssr_meteofrance')
    # ----------------------------- missing timestep -----------------------------
    # to check if there is station with too few records at daytime hours
    ssr_MeteoFrance_df['Hour'] = ssr_MeteoFrance_df.index.hour
    df_0 = ssr_MeteoFrance_df[['Hour', 'station_id']]
    df = df_0.pivot_table(index='station_id', columns='Hour', aggfunc=np.count_nonzero)
    df_2 = df_0.pivot_table(columns='station_id', index='Hour', aggfunc=np.count_nonzero)

if mean:
    ssr_MeteoFrance = f'SELECT station_id, dt as DateTime, ssr_meteofrance as ssr,' \
                      f'altitude, longitude, latitude ' \
                      f'from SWIO.SSR_Hourly_MeteoFrance_WRF35_WRF41_SARAH_E ' \
                      f'where dt>="2004-11-01" and dt<="2005-04-30 23:59:00" and ' \
                      f'hour(dt)>=8 and hour(dt)<=17;'

    ssr_MeteoFrance_df = GEO_PLOT.query_data(mysql_query=ssr_MeteoFrance)

    ssr_station_mean = ssr_MeteoFrance_df.groupby('station_id').mean()
    GEO_PLOT.plot_station_value(lon=ssr_station_mean['longitude'], lat=ssr_station_mean['latitude'],
                                value=ssr_station_mean['ssr'], cbar_label=r'SSR $(W/m^2)$',
                                fig_title='MEAN of MeteoFrance (8h00 ~ 17h00)')

    # NOTE: all the model data is selected by lon/lat index, defined as in the code
    # ~/Microsoft_OneDrive/OneDrive/CODE/Morel_2020/local_data/4.1_wrf/netcdf2csv.py,
    # according to the meteofrance stations' lon/lat.
    # TODO: rerun wrf model for 1 timestep, then add the xlong and xlat to the addout* files.

    GEO_PLOT.vis_a_vis_plot(x=ssr_station_mean['altitude'], y=ssr_station_mean['ssr'],
                            xlabel='altitude', ylabel='ssr_MeteoFrance', title='SSR vs altitude')

    # ----------------------------- WRF 4.1 -----------------------------
    ssr_wrf41 = f'SELECT station_id, dt as DateTime, swdown_wrf41 as ssr, longitude, latitude ' \
                f'from SWIO.SSR_Hourly_MeteoFrance_WRF35_WRF41_SARAH_E ' \
                f'where dt>="2004-11-01" and dt<="2005-04-30 23:59:00" and ' \
                f'hour(dt)>=8 and hour(dt)<=17;'

    ssr_wrf41_df = GEO_PLOT.query_data(mysql_query=ssr_wrf41)
    ssr_station_mean = ssr_wrf41_df.groupby('station_id').mean()
    GEO_PLOT.plot_station_value(lon=ssr_station_mean['longitude'], lat=ssr_station_mean['latitude'],
                                value=ssr_station_mean['ssr'], cbar_label=r'SSR $(W/m^2)$',
                                fig_title='MEAN of wrf4.1 (8h00 ~ 17h00)')

    # ----------------------------- WRF 3.5 -----------------------------
    ssr_wrf35 = f'SELECT station_id, dt as DateTime, swdown_wrf35 as ssr, longitude, latitude ' \
                f'from SWIO.SSR_Hourly_MeteoFrance_WRF35_WRF41_SARAH_E ' \
                f'where dt>="2004-11-01" and dt<="2005-04-30 23:59:00" and ' \
                f'hour(dt)>=8 and hour(dt)<=17;'

    ssr_wrf35_df = GEO_PLOT.query_data(mysql_query=ssr_wrf35)
    ssr_station_mean = ssr_wrf35_df.groupby('station_id').mean()
    GEO_PLOT.plot_station_value(lon=ssr_station_mean['longitude'], lat=ssr_station_mean['latitude'],
                                value=ssr_station_mean['ssr'], cbar_label=r'SSR $(W/m^2)$',
                                fig_title='MEAN of wrf3.5 (8h00 ~ 17h00)')

    # ----------------------------- WRF 4.1 - WRF 3.5 -----------------------------
if wrf41_vs_wrf35:

    ssr_wrf4135 = f'SELECT station_id, dt as DateTime, swdown_wrf35 as ssr35, swdown_wrf41 as ssr41, ' \
                  f'longitude, latitude, altitude ' \
                  f'from SWIO.SSR_Hourly_MeteoFrance_WRF35_WRF41_SARAH_E ' \
                  f'where dt>="2004-11-01" and dt<="2005-04-30 23:59:00" and ' \
                  f'hour(dt)>=8 and hour(dt)<=17;'

    ssr_wrf4135_df = GEO_PLOT.query_data(mysql_query=ssr_wrf4135)
    ssr_station_mean = ssr_wrf4135_df.groupby('station_id').mean()

    ssr_station_mean['diff_abs'] = np.abs(ssr_station_mean['ssr41'] - ssr_station_mean['ssr35'])
    ssr_station_mean['diff'] = ssr_station_mean['ssr41'] - ssr_station_mean['ssr35']

    # plot the difference
    GEO_PLOT.plot_station_value(lon=ssr_station_mean['longitude'], lat=ssr_station_mean['latitude'],
                                value=ssr_station_mean['diff'],
                                cbar_label=r'SSR $(W/m^2)$',
                                fig_title='abs(WRF4.1 - WRF3.5) (8h00 ~ 17h00)')

    GEO_PLOT.plot_station_value(lon=ssr_station_mean['longitude'], lat=ssr_station_mean['latitude'],
                                value=ssr_station_mean['diff_abs'],
                                cbar_label=r'SSR $(W/m^2)$',
                                fig_title='WRF4.1 - WRF3.5 (8h00 ~ 17h00)')

    # plot abs(diff) vs altitude:

    GEO_PLOT.vis_a_vis_plot(x=ssr_station_mean.altitude, y=ssr_station_mean.diff_abs,
                            xlabel='altitude', ylabel='abs(WRF4.1-WRF3.5)',
                            title='absolute (WRF4.1 - WRF3.5) vs station altitude')


# ----------------------------- plot diurnal cycle -----------------------------
if diurnal:

    ssr_diurnal = f'SELECT station_id, dt as DateTime, ssr_meteofrance, ssr_sarah_e, swdown_wrf41, swdown_wrf35 ' \
                  f'FROM SWIO.SSR_Hourly_MeteoFrance_WRF35_WRF41_SARAH_E ' \
                  f'where dt>="2004-11-01" and dt<="2005-04-30 23:59:00";'

    diurnal_data = GEO_PLOT.query_data(ssr_diurnal)
    GEO_PLOT.plot_hourly_curve_by_month(diurnal_data.groupby(diurnal_data.index).mean(),
                                        columns=['ssr_meteofrance', 'ssr_sarah_e', 'swdown_wrf41', 'swdown_wrf35'])

    # GEO_PLOT.plot_hourly_boxplot_by(diurnal_data, ['ssr_meteofrance', 'ssr_sarah_e', 'swdown_wrf41', 'swdown_wrf35'],
    #                                 by=None)
    GEO_PLOT.plot_hourly_boxplot_by(diurnal_data, ['ssr_meteofrance', 'ssr_sarah_e', 'swdown_wrf41', 'swdown_wrf35'],
                                    by='Month')

# ----------------------------- diurnal_bias -----------------------------
if diurnal_bias:

    ssr_diurnal = f'SELECT station_id, dt as DateTime, ssr_meteofrance, ssr_sarah_e, swdown_wrf41, swdown_wrf35 ' \
                  f'FROM SWIO.SSR_Hourly_MeteoFrance_WRF35_WRF41_SARAH_E ' \
                  f'where dt>="2004-11-01" and dt<="2005-04-30 23:59:00";'

    diurnal_data = GEO_PLOT.query_data(ssr_diurnal)

    diurnal_data['bias'] = diurnal_data['swdown_wrf41'] - diurnal_data['ssr_meteofrance']

    GEO_PLOT.plot_hourly_curve_by_month(diurnal_data.groupby(diurnal_data.index).mean(),
                                        columns=['bias'], bias=True)
# ----------------------------- plot statistics @ stations -----------------------------
if statistics:

    ssr_daytime = f'SELECT station_id, dt as DateTime,' \
                  f'ssr_meteofrance, ssr_sarah_e, swdown_wrf35, swdown_wrf41, longitude, latitude ' \
                  f'FROM SWIO.SSR_Hourly_MeteoFrance_WRF35_WRF41_SARAH_E ' \
                  f'where dt>="2004-11-01" and dt<="2005-04-30 23:59:00" and ' \
                  f'hour(dt)>=8 and hour(dt)<=17;'

    ssr_daytime_df = GEO_PLOT.query_data(ssr_daytime)

    station_group = ssr_daytime_df.groupby('station_id')
    statistics: pd.Series = station_group.apply(get_statistics).reset_index()
    lon = station_group[['longitude']].mean().values[:, 0]
    lat = station_group[['latitude']].mean().values[:, 0]

    values = [statistics[x] for x in statistics.columns[1:]]
    cbar_labels = [r'SSR $(W/m^2)$' for i in range(len(statistics.columns[1:]))]
    fig_titles = [s + ' (WRR4.1 and MétéoFrance) 8h00 ~ 17h00' for s in statistics.columns[1:]]
    bias_label = [True, False, False, False, False]

    for value, cbar_label, fig_title, bias_label in zip(values, cbar_labels, fig_titles, bias_label):
        GEO_PLOT.plot_station_value(lon=lon, lat=lat, value=value, cbar_label=cbar_label,
                                    fig_title=fig_title, bias=bias_label)

# ----------------------------- station with min/max bias -----------------------------


# ----------------------------- bias distribution -----------------------------

if station_mean_hourly_bias:
    # df: index=datetime, columns={'ssr_meteofrance', 'ssr_wrf35', 'ssr_sarah_e', 'month', 'zenith'}

    ssr_hourly = f'SELECT station_id, dt as DateTime, hour(dt) as Hour, longitude, latitude, ' \
                 f'ssr_wrf35, ssr_meteofrance, altitude, ' \
                 f'if(hour(dt)<12, zenith * (-1), zenith) as zenith ' \
                 f'FROM SWIO.SSR_Hourly_MeteoFrance_and_WRF35_and_SARAH_E ' \
                 f'where dt>="2004-11-01" and dt<="2005-04-30 23:59:00" and ' \
                 f'hour(dt)>=5 and hour(dt)<=20;'

    df: pd.DataFrame = GEO_PLOT.query_data(ssr_hourly)[:]

    # ----------------------------- save solar zenith to DataBase -----------------------------
    # df = GEO_PLOT.get_zenith_angle(df=df, datetime_col='DateTime', utc=False, lon=df['longitude'],
    #                                lat=df['latitude'], column_zenith='zenith')
    # NOTE: altitude is not included in the calculation of zenith angle !!!
    # ----------------------------- save solar zenith to DataBase -----------------------------

    df['radio'] = df['ssr_wrf35']/df['ssr_meteofrance']
    df = df.dropna()

    fig, ax = GEO_PLOT.plot_scatter_color_by(x=df['zenith'], y=df['radio'],
                                             label_x='Solar Zenith Angle', label_y='WRF35/MeteoFrance',
                                             color_by_column='Month')
    # some changes to the plot
    ax.set_ylim([0, 50])
    ax.set_xlim([-100, 100])

    plt.yscale('log', basey=2)
    ax.set_ylim(ymin=0.05)

    plt.title('SSR WRF3.5/MeteoFrance')

    plt.axhline(2, color='black', lw=2, linestyle=':')
    plt.axhline(1, color='black', lw=2)
    plt.axhline(0.5, color='black', lw=2, linestyle=':')

    print("waiting for the plot...")
    plt.show()

    # # solar zenith angle is saved to DataBase already.

if bias_distribution:
    bias_height = f'SELECT station_id, dt as DateTime, avg(altitude) as altitude, ' \
                  f'AVG(ABS(ssr_wrf35 - ssr_meteofrance)) as MAE ' \
                  f'FROM SWIO.SSR_Hourly_MeteoFrance_and_WRF35_and_SARAH_E ' \
                  f'where dt>="2004-11-01" and dt<="2005-04-30 23:59:00" and ' \
                  f'hour(dt)>=5 and hour(dt)<=20 ' \
                  f'group by station_id;'

    df: pd.DataFrame = GEO_PLOT.query_data(bias_height)

    # df['MAE'] = np.abs(df['ssr_wrf35'] - df['ssr_meteofrance']).dropna()

    fig, ax = GEO_PLOT.plot_scatter_color_by(x=df['altitude'], y=df['MAE'],
                                             label_x='altitude', label_y='MAE (WRF35-MeteoFrance)',
                                             color_by_column=None, size=6)

    plt.title('SSR: MAE with MeteoFrance')
    # plt.yscale('log')
    # plt.xscale('log')

    plt.show()
    print(f'got data')

if ssr_altitude:
    # to find the model accuracy with  altitude:

    bias_height = f'SELECT station_id, dt as DateTime, altitude, ' \
                  f'VARIANCE(ssr_meteofrance) as ssr_mf, ' \
                  f'VARIANCE(swdown_wrf35) as ssr_wrf35, ' \
                  f'VARIANCE(swdown_wrf41) as ssr_wrf41 ' \
                  f'FROM SWIO.SSR_Hourly_MeteoFrance_WRF35_WRF41_SARAH_E ' \
                  f'where dt>="2004-11-01" and dt<="2005-04-30 23:59:00" and ' \
                  f'hour(dt)>=5 and hour(dt)<=20 ' \
                  f'group by station_id;'

    bias_height_df: pd.DataFrame = GEO_PLOT.query_data(bias_height)

    bias_height_df['bias'] = bias_height_df['wrf41'] - bias_height_df['ssr_mf']

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(18, 6), facecolor='w', edgecolor='k', dpi=200)  # figsize=(w,h)
    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, hspace=0.4, top=0.8, wspace=0.05)

    plt.scatter(df['altitude'], df['ssr_wrf35'], marker='^', c='b', s=88, edgecolors='blue', alpha=0.8, label='WRF_3.5')
    plt.scatter(df['altitude'], df['ssr_wrf41'], marker='s', c='green', s=98, edgecolors='green', alpha=0.8, label='WRF_4.1')
    plt.scatter(df['altitude'], df['ssr_mf'], marker='o', c='red', s=58, edgecolors='red', alpha=0.8, label='MeteoFrance')

    plt.title('SSR vs altitude')
    plt.legend(loc="upper right", markerscale=1, fontsize=16)

    ax.set_xlabel('altitude (m)', fontsize=18)
    ax.set_ylabel('SSR (W/m2)', fontsize=18)
    ax.tick_params(labelsize=16)

    plt.grid(True)

    plt.show()

if hourly_bias_at_different_stations:
    # to select typical stations for deeper study.

    ssr_height = f'SELECT station_id, dt as DateTime, altitude, ' \
                 f'AVG(ssr_meteofrance) as ssr_mf, ' \
                 f'AVG(swdown_wrf35) as ssr_wrf35, ' \
                 f'AVG(swdown_wrf41) as ssr_wrf41 ' \
                 f'FROM SWIO.SSR_Hourly_MeteoFrance_WRF35_WRF41_SARAH_E ' \
                 f'where dt>="2004-11-01" and dt<="2005-04-30 23:59:00" and ' \
                 f'hour(dt)>=5 and hour(dt)<=20 ' \
                 f'group by station_id, hour(dt);'

    df: pd.DataFrame = GEO_PLOT.query_data(ssr_height)

    stations = df['station_id'].drop_duplicates()

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(18, 35), facecolor='w', edgecolor='k', dpi=200)  # figsize=(w,h)
    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, hspace=0.4, top=0.8, wspace=0.05)

    colors = GEO_PLOT.get_random_color(len(stations))

    for i in range(len(stations)):

        station = df[df.station_id == stations[i]]
        height = np.int(station.altitude.drop_duplicates())

        # plt.plot(station.index.hour, station.ssr_mf, color=colors[i], linestyle=':',
        #          label=str(stations[i]) + f' meteofrance ({height:g}m)')
        # plt.plot(station.index.hour, station.ssr_wrf41, color=colors[i], linestyle='-.',
        #          label=str(stations[i]) + f' wrf4.1 ({height:g}m)')
        # plt.plot(station.index.hour, station.ssr_wrf35, color=colors[i], linestyle='--',
        #          label=str(stations[i]) + f' wrf3.5 ({height:g}m)')

        # plt.plot(station.index.hour, station.ssr_wrf41 - station.ssr_wrf35, color=colors[i], linestyle='-',
        #          label=str(stations[i]) + f' wrf4.1-wrf3.5 ({height:g}m)')
                   # label=f'{height:g}m')
        # plt.plot(station.index.hour, station.ssr_wrf35 - station.ssr_mf, color=colors[i], linestyle='-',
        #          label=str(stations[i]) + f' wrf3.5-mf ({height:g}m)')

        plt.plot(station.index.hour, station.ssr_wrf41 - station.ssr_mf, color=colors[i], linestyle='-',
                 # label=str(stations[i]) + f' wrf4.1-mf ({height:g}m)')
                 label=f'{height:g}m')

    labelLines(plt.gca().get_lines(), fontsize=14)

    plt.xlabel('Hour', fontsize=13)
    plt.ylabel('SSR (w/m2)', fontsize=13)

    plt.grid()
    plt.legend()
    plt.show()

    print(f'got data')
# ----------------------------- monthly_mean_bias_at_station -----------------------------
if monthly_mean_bias_at_station:

    ssr_daytime = f'SELECT station_id, dt as DateTime,' \
                  f'ssr_meteofrance, ssr_sarah_e, swdown_wrf35, swdown_wrf41, longitude, latitude ' \
                  f'FROM SWIO.SSR_Hourly_MeteoFrance_WRF35_WRF41_SARAH_E ' \
                  f'where dt>="2004-11-01" and dt<="2005-04-30 23:59:00" and ' \
                  f'hour(dt)>=8 and hour(dt)<=17;'

    ssr_daytime_df = GEO_PLOT.query_data(ssr_daytime, remove_missing_data=True)

    station_group = ssr_daytime_df.groupby('station_id')
    lon = station_group[['longitude']].mean().values[:, 0]
    lat = station_group[['latitude']].mean().values[:, 0]

    ssr_daytime_df['bias'] = ssr_daytime_df['swdown_wrf41'] - ssr_daytime_df['ssr_meteofrance']

    # ----------------------------- plotting -----------------------------
    GEO_PLOT.plot_station_value_by_month(lon, lat, ssr_daytime_df[['station_id', 'bias']],
                                         cbar_label=r'SSR $(W/m^2)$', fig_title='test', bias=True)


if monthly_circulation_era5:

    wind_era5 = f'{DIR:s}/data/ERA5/u.v.850hPa.era5.monthly.swio.nc'
    wind_ds = xr.open_dataset(wind_era5)

    sea_level_pressure = f'{DIR:s}/data/ERA5/msl.t2m.ssrd.era5.monthly.swio.nc'

    ds = xr.open_dataset(sea_level_pressure)
    press_ds = ds.msl
    ssr = ds.ssrd/85400

    # ----------------------------- plotting -----------------------------
    GEO_PLOT.monthly_circulation(lon=wind_ds.longitude, lat=wind_ds.latitude,
                                 u=wind_ds.u, v=wind_ds.v, p=ssr, domain='reu_mau',
                                 cbar_label='mean sea level pressure', fig_title='testing', bias=False)

    print(f'ggg')


# ==================================================================== cloud
if where_and_when_cloud:

    # bias
    ssr_daytime = f'SELECT station_id, dt as DateTime,' \
                  f'ssr_meteofrance, ssr_sarah_e, swdown_wrf35, swdown_wrf41, longitude, latitude ' \
                  f'FROM SWIO.SSR_Hourly_MeteoFrance_WRF35_WRF41_SARAH_E ' \
                  f'where dt>="2004-12-01" and dt<="2005-02-28 23:59:00" and ' \
                  f'hour(dt)>=8 and hour(dt)<=17;'

    ssr_daytime_df = GEO_PLOT.query_data(ssr_daytime)

    station_group = ssr_daytime_df.groupby('station_id')
    lon = station_group[['longitude']].mean().values[:, 0]
    lat = station_group[['latitude']].mean().values[:, 0]

    ssr_daytime_df['bias'] = ssr_daytime_df['swdown_wrf41'] - ssr_daytime_df['ssr_meteofrance']

    # cloud fraction cover
    cfc_cmsaf = f'{DIR:s}/local_data/obs/CFC.cmsaf.hour.reu.DJF.nc'
    cloud = xr.open_dataset(cfc_cmsaf)
    print(f'good data')
    # ----------------------------- plotting -----------------------------

    # plot diurnal cloud and ssr bias:
    GEO_PLOT.plot_scatter_contourf(lon=cloud.lon, lat=cloud.lat, cloud=cloud.CFC, cbar_label='mean cloud fraction (%)',
                                   lon_mf=lon, lat_mf=lat, value=ssr_daytime_df[['bias', 'station_id']],
                                   cbar_mf=r'SSR WRF4.1-MF $(W/m^2)$',
                                   bias=False, bias_mf=True)


if monthly_mean_bias_clustering:


    # bias
    ssr_daytime = f'SELECT station_id, dt as DateTime,' \
                  f'ssr_meteofrance, ssr_sarah_e, swdown_wrf35, swdown_wrf41, longitude, latitude ' \
                  f'FROM SWIO.SSR_Hourly_MeteoFrance_WRF35_WRF41_SARAH_E ' \
                  f'where dt>="2004-12-01" and dt<="2005-02-28 23:59:00" and ' \
                  f'hour(dt)>=8 and hour(dt)<=17;'

    df = GEO_PLOT.query_data(ssr_daytime)

    df['bias'] = df['swdown_wrf41'] - df['ssr_meteofrance']

    stations = list(set(df.station_id))

    DJF_mean_hourly_bias = pd.DataFrame(columns=[str(i) for i in range(8, 18)])

    for i in range(len(stations)):
        sta = df[df['station_id'] == stations[i]]
        sta['hour'] = sta.index.hour
        sta_hour_group = sta.groupby('hour').mean()

        temp = sta_hour_group['bias'].values.reshape(1, 10)

        temp_df = pd.DataFrame(columns=[str(i) for i in range(8, 18)], data=temp)
        DJF_mean_hourly_bias = DJF_mean_hourly_bias.append(temp_df)

    DJF_mean_hourly_bias['station_id'] = stations
    DJF_mean_hourly_bias = DJF_mean_hourly_bias.set_index('station_id', drop=True)

    # locations
    locations = df.groupby('station_id').mean()[['longitude', 'latitude']]

    # ----------------------------- clustering -----------------------------
    bias_cluster_mean, labels = GEO_PLOT.cluster_mean_gaussian_mixture(DJF_mean_hourly_bias, n_components=2,
                                                                       max_iter=100, cov_type='diag')

    GEO_PLOT.plot_daily_cluster_mean(bias_cluster_mean, locations, labels, ylabel='WRF4.1-MF',
                                     title='DJF_mean hourly clustering of bias')







































