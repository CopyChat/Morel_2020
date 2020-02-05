"""
to validation and analysis WRF over la Reunion.
"""

__version__ = f'Version 1.0  \nTime-stamp: <2019-09-24>'
__author__ = "ChaoTANG@univ-reunion.fr"

import sys
import numpy as np
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

    bias = np.mean(gg['ssr_wrf35'] - gg['ssr_meteofrance'])
    r2 = r2_score(gg['ssr_wrf35'], gg['ssr_meteofrance'])
    mae = mean_absolute_error(gg['ssr_wrf35'], gg['ssr_meteofrance'])
    rmse = np.sqrt(mean_squared_error(gg['ssr_wrf35'], gg['ssr_meteofrance']))
    corr = np.corrcoef(gg['ssr_wrf35'], gg['ssr_meteofrance'])[0, 1]

    return pd.Series(dict(MEAN_BIAS=bias, R2=r2, MAE=mae, RMSE=rmse, corr=corr))


# ----------------------------- plot station mean -----------------------------
bias = 0
mean = 1
diurnal = 0
statistics = 0
bias_distribution = 0
ssr_altitude = 0
hourly_bias_at_different_stations = 0

if mean:
    ssr_MeteoFrance = f'SELECT station_id, dt as DateTime, swdown_wrf41 as ssr, longitude, latitude ' \
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
# ----------------------------- plot diurnal cycle -----------------------------

if diurnal:

    ssr_diurnal = f'SELECT station_id, dt as DateTime,' \
                  f'ssr_meteofrance, ssr_sarah_e, ssr_wrf35 ' \
                  f'FROM SWIO.SSR_Hourly_MeteoFrance_and_WRF35_and_SARAH_E ' \
                  f'where dt>="2004-11-01" and dt<="2005-04-30 23:59:00";'

    diurnal_data = GEO_PLOT.query_data(ssr_diurnal)
    GEO_PLOT.plot_hourly_curve_by_month(diurnal_data.groupby(diurnal_data.index).mean(),
                                        columns=['ssr_meteofrance', 'ssr_sarah_e', 'ssr_wrf35'])

    GEO_PLOT.plot_hourly_boxplot_by(diurnal_data, ['ssr_meteofrance', 'ssr_sarah_e', 'ssr_wrf35'], by=None)
    GEO_PLOT.plot_hourly_boxplot_by(diurnal_data, ['ssr_meteofrance', 'ssr_sarah_e', 'ssr_wrf35'], by='Month')

# ----------------------------- plot statistics @ stations -----------------------------
if statistics:

    ssr_daytime = f'SELECT station_id, dt as DateTime,' \
                  f'ssr_meteofrance, ssr_sarah_e, ssr_wrf35, longitude, latitude ' \
                  f'FROM SWIO.SSR_Hourly_MeteoFrance_and_WRF35_and_SARAH_E ' \
                  f'where dt>="2004-11-01" and dt<="2005-04-30 23:59:00" and ' \
                  f'hour(dt)>=8 and hour(dt)<=17;'

    ssr_daytime_df = GEO_PLOT.query_data(ssr_daytime)

    station_group = ssr_daytime_df.groupby('station_id')
    statistics: pd.Series = station_group.apply(get_statistics).reset_index()
    lon = station_group[['longitude']].mean().values[:, 0]
    lat = station_group[['latitude']].mean().values[:, 0]

    values = [statistics[x] for x in statistics.columns[1:]]
    cbar_labels = [r'SSR $(W/m^2)$' for i in range(len(statistics.columns[1:]))]
    fig_titles = [s + ' (WRF3.5 and MétéoFrance) 8h00 ~ 17h00' for s in statistics.columns[1:]]

    for value, cbar_label, fig_title in zip(values, cbar_labels, fig_titles):
        GEO_PLOT.plot_station_value(lon=lon, lat=lat, value=value, cbar_label=cbar_label, fig_title=fig_title)

# ----------------------------- station with min/max bias -----------------------------


# ----------------------------- bias distribution -----------------------------

if bias:
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

    ssr_height = f'SELECT station_id, dt as DateTime, altitude, ' \
                 f'AVG(ssr_meteofrance) as ssr_mf, ' \
                 f'AVG(swdown_wrf35) as ssr_wrf35, ' \
                 f'AVG(swdown_wrf41) as ssr_wrf41 ' \
                 f'FROM SWIO.SSR_Hourly_MeteoFrance_WRF35_WRF41_SARAH_E ' \
                 f'where dt>="2004-11-01" and dt<="2005-04-30 23:59:00" and ' \
                 f'hour(dt)>=5 and hour(dt)<=20 ' \
                 f'group by station_id;'

    ssr_height = f'SELECT station_id, dt as DateTime, altitude, ' \
                 f'VARIANCE(ssr_meteofrance) as ssr_mf, ' \
                 f'VARIANCE(swdown_wrf35) as ssr_wrf35, ' \
                 f'VARIANCE(swdown_wrf41) as ssr_wrf41 ' \
                 f'FROM SWIO.SSR_Hourly_MeteoFrance_WRF35_WRF41_SARAH_E ' \
                 f'where dt>="2004-11-01" and dt<="2005-04-30 23:59:00" and ' \
                 f'hour(dt)>=5 and hour(dt)<=20 ' \
                 f'group by station_id;'

    df: pd.DataFrame = GEO_PLOT.query_data(ssr_height)

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
