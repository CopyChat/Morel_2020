"""
functions used for get and plot geo data
"""

__version__ = f'Version 1.0  \nTime-stamp: <2019-02-21>'
__author__ = "ChaoTANG@univ-reunion.fr"

import pandas as pd
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

DIR = f'/Users/ctang/Microsoft_OneDrive/OneDrive/CODE/Morel_2020'


def find_two_bounds(min: float, max: float, n: int):
    """
    find the vmin and vmax in 'n' interval
    :param min:
    :param max:
    :param n:
    :return:
    """
    left = round(min / n, 0) * n
    right = round(max / n, 0) * n

    return left, right


def query_data(mysql_query: str, remove_missing_data=True):
    """
    select data from DataBase
    :return: DataFrame
    """

    from sqlalchemy import create_engine
    import pymysql
    pymysql.install_as_MySQLdb()

    db_connection_str = 'mysql+pymysql://pepapig:123456@localhost/SWIO'
    db_connection = create_engine(db_connection_str)

    df: pd.DataFrame = pd.read_sql(sql=mysql_query, con=db_connection)

    # ----------------------------- remove two stations with many missing data -----------------------------
    if remove_missing_data:
        df.drop(df[df['station_id'] == 97419380].index, inplace=True)
        df.drop(df[df['station_id'] == 97412384].index, inplace=True)

    # ----------------------------- remove two stations with many missing data -----------------------------
    df = df.set_index('DateTime')

    return df


def plot_station_value(lon: pd.DataFrame, lat: pd.DataFrame, value: np.array, cbar_label: str,
                       fig_title: str, bias=False):
    """
    plot station locations and their values
    :param fig_title:
    :param cbar_min:
    :param cbar_max:
    :param cbar_label: label of color bar
    :param lon:
    :param lat:
    :param value:
    :return: map show
    """
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import matplotlib.pyplot as plt
    import matplotlib as mpl

    fig = plt.figure(dpi=220)
    fig.suptitle(fig_title)

    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax.set_extent([55, 56, -21.5, -20.8], crs=ccrs.PlateCarree())
    # lon_left, lon_right, lat_north, lat_north

    ax.add_feature(cfeature.LAND.with_scale('10m'))
    # ax.add_feature(cfeature.OCEAN.with_scale('10m'))
    # ax.add_feature(cfeature.COASTLINE.with_scale('10m'))
    # ax.add_feature(cfeature.BORDERS.with_scale('10m'), linestyle=':')
    # ax.add_feature(cfeature.LAKES.with_scale('10m'), alpha=0.5)
    # ax.add_feature(cfeature.RIVERS.with_scale('10m'))

    # ax.coastlines()

    # ----------------------------- stations -----------------------------

    if np.max(value) - np.min(value) < 10:
        round_number = 2
    else:
        round_number = 0

    n_cbar = 10
    vmin = round(np.min(value) / n_cbar, round_number) * n_cbar
    vmax = round(np.max(value) / n_cbar, round_number) * n_cbar

    if bias:
        cmap = plt.cm.coolwarm
        vmin = max(np.abs(vmin), np.abs(vmax)) * (- 1)
        vmax = max(np.abs(vmin), np.abs(vmax))
    else:
        cmap = plt.cm.YlOrRd

    bounds = np.linspace(vmin, vmax, n_cbar + 1)
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    # ----------------------------------------------------------
    sc = plt.scatter(lon, lat, c=value, edgecolor='black',
                     # transform=ccrs.PlateCarree(),
                     zorder=2, norm=norm, vmin=vmin, vmax=vmax, s=50, cmap=cmap)

    # ----------------------------- color bar -----------------------------
    cb = plt.colorbar(sc, orientation='horizontal', shrink=0.7, pad=0.05, label=cbar_label)
    cb.ax.tick_params(labelsize=10)

    ax.gridlines(draw_labels=True)
    plt.tight_layout(pad=0.1, w_pad=0.3, h_pad=0.6)

    plt.show()
    print(f'got plot')


def get_zenith_angle(df: pd.DataFrame, datetime_col: str, utc: bool,
                     lon: np.ndarray, lat: np.ndarray, column_zenith: str):
    """
    get solar zenith angle at (lon, lat) according to df with DateTimeUTC
    :param column_zenith:
    :param utc:
    :param datetime_col:
    :param df:
    :param lon:
    :param lat:
    :return:
    """

    import pytz
    from pysolar.solar import get_altitude

    if utc:
        df['DateTimeUTC_2'] = df[datetime_col]
    else:
        # add timezone info, which is needed by pysolar
        df['DateTimeUTC_2'] = [df.index.to_pydatetime()[i].astimezone(pytz.timezone("UTC"))
                               for i in range(len(df.index))]

    print(f'starting calculation solar zenith angle')
    zenith = [90 - get_altitude(lat[i], lon[i], df['DateTimeUTC_2'][i]) for i in range(len(df))]
    # prime meridian in Greenwich, England

    # df_new = df.copy()
    # df_new['utc'] = df['DateTimeUTC_2']
    # df_new[column_zenith] = zenith
    df_new = pd.DataFrame(columns=[column_zenith], index=df.index, data=zenith)

    output_file = r'/Users/ctang/Microsoft_OneDrive/OneDrive/CODE/Prediction_PV/zenith.csv'
    df_new.to_csv(output_file)

    # ----------------------------- for test -----------------------------
    # import datetime
    # date = datetime.datetime(2004, 11, 1, 00, 00, 00, tzinfo=datetime.timezone.utc)
    #
    # for lat in range(100):
    #     lat2 = lat/100 - 21
    #     a = 90 - get_altitude(55.5, lat2, date)
    #     print(lat2, a)
    # ----------------------------- for test -----------------------------

    return df_new


def zenith_angle_reunion(df, ):
    """
    to get zenith angle @ la reunion
    input: df['DateTime']
    """
    from pysolar.solar import get_altitude

    lat = -22  # positive in the northern hemisphere
    lon = 55  # negative reckoning west from
    # prime meridian in Greenwich, England

    return [90 - get_altitude(lat, lon, df[i])
            for i in range(len(df))]


def get_color():
    """define some (8) colors to use for plotting ... """

    # return [plt.cm.Spectral(each)
    #         for each in np.linspace(0, 6, 8)]

    # return ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    return ['pink', 'darkviolet', 'blue', 'teal', 'forestgreen', 'darkorange', 'red',
            'deeppink', 'blueviolet', 'royalblue', 'lightseagreen', 'limegreen', 'yellowgreen', 'tomato',
            'silver', 'gray', 'black']


def plot_hourly_curve_by_month(df: pd.DataFrame, columns: list, suptitle=' ', bias=False):
    """
    plot hourly curves by /month/ for the columns in list
    :param suptitle:
    :param df:
    :param columns:
    :return:
    """

    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    # ----------------------------- set parameters -----------------------------
    months = [11, 12, 1, 2, 3, 4]
    colors = ['black', 'green', 'orange', 'red']
    data_sources = ['MeteoFrance', 'SARAH-E', 'WRF4.1', 'WRF3.5']

    # ----------------------------- set fig -----------------------------
    nrows = len(months)

    fig, axs = plt.subplots(nrows=nrows, ncols=1, figsize=(16, 16),
                            facecolor='w', edgecolor='k', dpi=200)  # figsize=(w,h)
    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, hspace=0.4, top=0.8, wspace=0.05)

    axs = axs.ravel()
    # ----------------------------- plotting -----------------------------

    from matplotlib.dates import DateFormatter
    for v in range(len(columns)):
        for i in range(nrows):
            ax = plt.sca(axs[i])  # active this subplot

            month = months[i]
            data_slice = df[df.index.month == month]
            x = range(len(data_slice))

            if bias:
                label = f'WRF4.1 - {data_sources[v]:s}'
            else:
                label = data_sources[v]

            plt.plot(x, data_slice[[columns[v]]], color=colors[v], label=label)

            print(f'month = {month:g}, var = {columns[v]:s}')

            # ----------------------------- format of fig -----------------------------
            nday = len(set(data_slice.index.day))

            custom_ticks = range(11, len(data_slice), 24)

            custom_ticks_labels = range(1, nday + 1)
            axs[i].set_xticks(custom_ticks)
            axs[i].set_xticklabels(custom_ticks_labels)

            axs[i].set_xlim(0, 24 * 34)

            # axs[i].xaxis.set_ticks_position('top')
            # axs[i].xaxis.set_ticks_position('bottom')

            plt.legend(loc='upper right', fontsize=8)
            plt.xlabel(f'day')
            plt.ylabel(r'$SSR\ (W/m^2)$')
            axs[i].text(0.5, 0.95, data_slice.index[0].month_name(),
                        horizontalalignment='right', verticalalignment='top', transform=axs[i].transAxes)

    plt.suptitle(suptitle)
    plt.show()
    print(f'got the plot')


def plot_hourly_boxplot_by(df: pd.DataFrame, columns: list, by: str):
    """
    plot hourly box plot by "Month" or "Season"
    :param df:
    :param columns:
    :param by:
    :return:
    """

    import seaborn
    import matplotlib.pyplot as plt

    if by == 'Month':
        nrow = 7
        ncol = 1
    if by is None:
        nrow = 1
        ncol = 1

    n_plot = nrow * ncol

    fig, axs = plt.subplots(nrows=nrow, ncols=ncol,
                            figsize=(10, 19), facecolor='w', edgecolor='k', dpi=200)  # figsize=(w,h)

    if by == 'Month':
        axs = axs.ravel()

    months = [11, 12, 1, 2, 3, 4, 4]
    for i in range(len(months)):
        if by == 'Month':
            # plt.sca(axs[i])  # active this subplot
            ax = axs[i]
        if by is None:
            ax = axs

        if by == 'Month':
            data_slice = df[df.index.month == months[i]]
        if by is None:
            data_slice = df.copy()

        all_var = pd.DataFrame()
        for col in range(len(columns)):
            # print(f'for var = {columns[col]:s}')
            # calculate normalised value:
            var = pd.DataFrame()
            var['target'] = data_slice[columns[col]]
            var['Hour'] = data_slice.index.hour
            var['var'] = [columns[col] for x in range(len(data_slice))]
            all_var = all_var.append(var)

        seaborn.boxplot(x='Hour', y='target', hue='var', data=all_var, ax=ax, showmeans=True)

        ax.set_xlim(5, 20)
        # ax.set_ylim(0, 1.1)
        if by is not None:
            ax.set_title(f'{by:s} = {months[i]:g}')

        # plt.legend()

        # Get the handles and labels. For this example it'll be 2 tuples
        # of length 4 each.
        handles, labels = ax.get_legend_handles_labels()

        # When creating the legend, only use the first two elements
        # to effectively remove the last two.
        plt.legend(handles[0:4], labels[0:4], bbox_to_anchor=(0.95, 0.95), borderaxespad=0.,
                   loc="upper right", fontsize=18)

        plt.ylabel(f'distribution')
        plt.title(f'SSR distribution between 5AM - 8PM', fontsize=18)
        ax.set_xlabel(f'Hour', fontsize=18)
        ax.set_ylabel(f'SSR ($W/m^2$)', fontsize=18)
        ax.tick_params(labelsize=16)

        ax.set_axisbelow(True)
        ax.yaxis.grid(color='gray', linestyle='dashed')

    print(f'save/show the plot ...')
    # plt.savefig(f'train_boxplot_by_{by:s}.png', dpi=200)

    plt.show()

    print(f'got this plot')


def plot_scatter_color_by(x: pd.DataFrame, y: pd.DataFrame, label_x: str, label_y: str,
                          color_by_column: str, size: float = 8):
    """

    :param size:
    :param x:
    :param y:
    :param label_x:
    :param label_y:
    :param color_by_column:
    :return:
    """
    import matplotlib.pyplot as plt

    # default is color_by = month

    months = [11, 12, 1, 2, 3, 4]
    month_names = ['Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr']

    colors = ['pink', 'darkviolet', 'blue', 'forestgreen', 'darkorange', 'red',
              'deeppink', 'blueviolet', 'royalblue', 'lightseagreen', 'limegreen', 'yellowgreen', 'tomato',
              'silver', 'gray', 'black']

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(18, 6), facecolor='w', edgecolor='k', dpi=200)  # figsize=(w,h)
    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.15, hspace=0.4, top=0.8, wspace=0.05)

    if color_by_column is None:
        xx = x
        yy = y
        plt.scatter(xx, yy, c=colors[1], s=size, edgecolors=colors[1], alpha=0.8)

    if color_by_column == 'Month':
        for i in range(len(months)):
            xx = x[x.index.month == months[i]]
            yy = y[y.index.month == months[i]]

            # plt.plot(xx, yy, label=month_names[i], color=colors[i])
            plt.scatter(xx, yy, c=colors[i], label=month_names[i],
                        s=size, edgecolors=colors[i], alpha=0.8)

        plt.legend(loc="upper right", markerscale=6, fontsize=16)

    ax.set_xlabel(label_x, fontsize=18)
    ax.set_ylabel(label_y, fontsize=18)
    ax.tick_params(labelsize=16)

    plt.grid(True)

    return fig, ax


# ==================================
def get_random_color(num_color: int):
    """
    return color as a list
    :param num_color:
    :return:
    """
    import matplotlib.pyplot as plt
    import random

    number_of_colors = num_color

    color = ["#" + ''.join([random.choice('0123456789ABCDEF') for j in range(6)])
             for i in range(number_of_colors)]

    return color


# ==================================
def station_data_missing_map_hourly_by_month(df: pd.DataFrame, station_id: str, columns: str):
    """
    plot hourly missing data map by month
    :param station_id:
    :param df:
    :param columns: which column to be checked
    :return:
    """

    import matplotlib.pyplot as plt
    # ----------------------------- set parameters -----------------------------
    # TODO: read month directly
    months = [11, 12, 1, 2, 3, 4]
    station_id = list(set(df[station_id]))
    # ----------------------------- set fig -----------------------------
    nrows = len(months)

    fig, axs = plt.subplots(nrows=nrows, ncols=1, figsize=(10, 30),
                            facecolor='w', edgecolor='k', dpi=200)  # figsize=(w,h)
    fig.subplots_adjust(left=0.15, right=0.95, bottom=0.05, hspace=0.4, top=0.96, wspace=0.05)

    axs = axs.ravel()
    # ----------------------------- plotting -----------------------------

    from matplotlib.dates import DateFormatter

    # plot data in each month:
    for i in range(len(months)):
        month = months[i]

        ax = plt.sca(axs[i])  # active this subplot

        month_data = df[df.index.month == month].sort_index()

        # get all time steps to check missing value
        first_timestep = month_data.index[0]
        last_timestep = month_data.index[-1]
        time_range = pd.date_range(first_timestep, last_timestep, freq='60min')
        daytime_range = [x for x in time_range if (8 <= x.hour <= 17)]
        all_daytime_index = pd.Index(daytime_range)

        # for v in range(2):
        for v in range(len(station_id)):

            data_slice = month_data[month_data['station_id'] == station_id[v]]
            nday = len(set(data_slice.index.day))

            print(f'month = {month:g}, station_id = {v:g}, day = {nday:g}')

            # find missing time steps:
            diff = all_daytime_index.difference(data_slice.index)

            if len(diff) == 0:
                print(f'all complete ...')
                plt.hlines(v, 0, 320, colors='blue', linewidth=0.2, linestyles='dashed', label='')
            else:
                print(f'there is missing data ...')

                plt.hlines(v, 0, 320, colors='red', linewidth=0.4, linestyles='dashed', label='')
                for k in range(len(all_daytime_index)):
                    if all_daytime_index[k] in diff:
                        plt.scatter(k, v, edgecolor='black', zorder=2, s=50)

        # ----------------------------- format of fig -----------------------------

        # ----------------------------- x axis -----------------------------
        # put the ticks in the middle of the day, means 12h00
        custom_ticks = range(4, len(data_slice), 10)

        custom_ticks_labels = range(1, nday + 1)
        axs[i].set_xticks(custom_ticks)
        axs[i].set_xticklabels(custom_ticks_labels)
        axs[i].set_xlim(0, 320)

        # axs[i].xaxis.set_ticks_position('top')
        # axs[i].xaxis.set_ticks_position('bottom')

        # ----------------------------- y axis -----------------------------
        custom_ticks = range(len(station_id))

        custom_ticks_labels = station_id

        axs[i].set_yticks(custom_ticks)
        axs[i].set_yticklabels(custom_ticks_labels)

        axs[i].set_ylim(-1, len(station_id) + 1)

        # plt.legend(loc='upper right', fontsize=8)
        plt.xlabel(f'day')
        plt.ylabel(f'station_id (blue (red) means (not) complete in this month)')
        plt.title(data_slice.index[0].month_name())

    suptitle = f'MeteoFrance missing data at each station during daytime (8h - 17h)'
    plt.suptitle(suptitle)

    # plt.show()
    print(f'got the plot')
    plt.savefig('./meteofrance_missing_map.png', dpi=200)


def plot_station_value_by_month(lon: pd.DataFrame, lat: pd.DataFrame, value: pd.DataFrame,
                                cbar_label: str, fig_title: str, bias=False):
    """
    plot station locations and their values
    :param bias:
    :param fig_title:
    :param cbar_label: label of color bar
    :param lon:
    :param lat:
    :param value:
    :return: map show
    """
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import matplotlib.pyplot as plt
    import matplotlib as mpl

    months = [11, 12, 1, 2, 3, 4]
    month_names = ['Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr']
    nrows = len(months)

    fig = plt.figure(figsize=(5, 24), dpi=200)
    # fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, hspace=1.5, top=0.95, wspace=0.05)

    # plot in each month:
    for m in range(len(months)):

        # data:
        monthly_data = value[value.index.month == months[m]]

        station_group = monthly_data.groupby('station_id')
        station_mean_bias = station_group[['bias']].mean().values[:, 0]

        # set map
        ax = plt.subplot(len(months), 1, m + 1, projection=ccrs.PlateCarree())
        ax.set_extent([55, 56, -21.5, -20.8], crs=ccrs.PlateCarree())
        # ax.set_extent([20, 110, -51, 9], crs=ccrs.PlateCarree())
        # lon_left, lon_right, lat_north, lat_north

        ax.coastlines('50m')
        ax.add_feature(cfeature.LAND.with_scale('10m'))
        # ax.add_feature(cfeature.OCEAN.with_scale('10m'))
        # ax.add_feature(cfeature.COASTLINE.with_scale('10m'))
        # ax.add_feature(cfeature.BORDERS.with_scale('10m'), linestyle=':')
        # ax.add_feature(cfeature.LAKES.with_scale('10m'), alpha=0.5)
        # ax.add_feature(cfeature.RIVERS.with_scale('10m'))
        # ax.coastlines()

        # ----------------------------- cbar -----------------------------
        if np.max(station_mean_bias) - np.min(station_mean_bias) < 10:
            round_number = 2
        else:
            round_number = 0

        n_cbar = 10
        vmin = round(np.min(station_mean_bias) / n_cbar, round_number) * n_cbar
        vmax = round(np.max(station_mean_bias) / n_cbar, round_number) * n_cbar

        if bias:
            cmap = plt.cm.coolwarm
            vmin = max(np.abs(vmin), np.abs(vmax)) * (- 1)
            vmax = max(np.abs(vmin), np.abs(vmax))
        else:
            cmap = plt.cm.YlOrRd

        vmin = -340
        vmax = 340

        bounds = np.linspace(vmin, vmax, n_cbar + 1)
        norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
        # ----------------------------------------------------------
        # plot:
        # ax.quiver(x, y, u, v, transform=vector_crs)
        sc = plt.scatter(lon, lat, c=station_mean_bias, edgecolor='black',
                         # transform=ccrs.PlateCarree(),
                         zorder=2, norm=norm, vmin=vmin, vmax=vmax, s=50, cmap=cmap)

        # ----------------------------- color bar -----------------------------
        cb = plt.colorbar(sc, orientation='horizontal', shrink=0.7, pad=0.1, label=cbar_label)
        cb.ax.tick_params(labelsize=10)

        # ax.xaxis.set_ticks_position('top')

        ax.gridlines(draw_labels=False)

        ax.text(0.93, 0.95, month_names[m],
                horizontalalignment='right', verticalalignment='top', transform=ax.transAxes)

    plt.tight_layout(pad=0.1, w_pad=0.3, h_pad=0.6)
    plt.suptitle(f'monthly daytime (8h 17h) \n mean bias at MeteoFrance stations')

    plt.show()
    print(f'got plot')


def monthly_circulation(lon: xr.DataArray, lat: xr.DataArray,
                        u: xr.DataArray, v: xr.DataArray, p: xr.DataArray, domain:str,
                        cbar_label: str, fig_title: str, bias=False):

    """
    to plot monthly circulation, u, v winds, and mean sea level pressure (p)
    :param domain: one of ['swio', 'reu-mau', 'reu']
    :param p:
    :param v:
    :param u:
    :param bias:
    :param fig_title:
    :param cbar_label: label of color bar
    :param lon:
    :param lat:
    :return: map show
    """

    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import matplotlib.pyplot as plt
    import matplotlib as mpl

    months = [11, 12, 1, 2, 3, 4]
    dates = ['2004-11-01', '2004-12-01', '2005-01-01', '2005-02-01', '2005-03-01', '2005-04-01']
    month_names = ['Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr']
    nrows = len(months)

    fig = plt.figure(figsize=(5, 24), dpi=300)
    # fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, hspace=1.5, top=0.95, wspace=0.05)

    # plot in each month:
    for m in range(len(months)):
        ax = plt.subplot(len(months), 1, m + 1, projection=ccrs.PlateCarree())

        # ax.gridlines(draw_labels=False)

        print(f'plot month = {month_names[m]:s}')
        # ----------------------------- plot u and v winds -----------------------------
        # data:
        x = lon.longitude.values
        y = lat.latitude.values
        monthly_u = u.sel(time=dates[m]).values
        monthly_v = v.sel(time=dates[m]).values
        monthly_p = p.sel(time=dates[m]).values

        # set map
        area_name = domain

        if area_name == 'swio':
            n_sclice = 1
            n_scale = 2
        if area_name == 'reu_mau':
            n_sclice = 1
            n_scale = 10

        if area_name == 'reu':
            n_sclice = 2
            n_scale = 10

        area = get_lonlatbox(area_name)
        ax.set_extent(area, crs=ccrs.PlateCarree())
        # lon_left, lon_right, lat_north, lat_north

        ax.coastlines('50m')
        ax.add_feature(cfeature.LAND.with_scale('10m'))

        # ----------------------------- mean sea level pressure -----------------------------
        # Contour the heights every 10 m
        contours = np.arange(98947, 102427, 300)

        c = ax.contour(x, y, monthly_p, levels=contours, colors='green', linewidths=1)
        ax.clabel(c, fontsize=10, inline=1, inline_spacing=3, fmt='%i')


        # ----------------------------- wind -----------------------------
        # Set up parameters for quiver plot. The slices below are used to subset the data (here
        # taking every 4th point in x and y). The quiver_kwargs are parameters to control the
        # appearance of the quiver so that they stay consistent between the calls.
        quiver_slices = slice(None, None, n_sclice)
        quiver_kwargs = {'headlength': 5, 'headwidth': 3, 'angles': 'uv', 'scale_units': 'xy', 'scale': n_scale}

        # Plot the wind vectors
        wind = ax.quiver(x[quiver_slices], y[quiver_slices],
                         monthly_u[quiver_slices, quiver_slices], monthly_v[quiver_slices, quiver_slices],
                         color='blue', zorder=2, **quiver_kwargs)

        ax.text(0.93, 0.95, month_names[m],
                horizontalalignment='right', verticalalignment='top', transform=ax.transAxes)

        # ax.gridlines(draw_labels=False)
        # # ----------------------------------------------------------
        # # plot:

        clevs = np.arange(-10, 12, 2)
        # clevs = np.arange(200, 370, 15)
        cf = ax.contourf(x, y, monthly_u, clevs, cmap=plt.cm.coolwarm,
                         norm=plt.Normalize(-10, 10), transform=ccrs.PlateCarree())

        # ----------------------------- color bar -----------------------------
        cb = plt.colorbar(cf, orientation='horizontal', shrink=0.7, pad=0.05,
                          # label='ssr')
                          label='east <-- 850hPa zonal wind --> west')

        cb.ax.tick_params(labelsize=10)

        # # ax.xaxis.set_ticks_position('top')

        # ax.text(0.53, 0.95, month_names[m] + 'from ERA5 2004-2005',
        #         horizontalalignment='right', verticalalignment='top',
        #         transform=ax.transAxes)

        plt.title(month_names[m] + ' (ERA5 2004-2005)')

    # plt.tight_layout(pad=0.1, w_pad=0.3, h_pad=0.6)
    plt.suptitle(f'circulation wind at 850 hPa')

    plt.show()
    print(f'got plot')

    plt.savefig(f'{DIR:s}/monthly_circulation.png', dpi=220)


def vis_a_vis_plot(x, y, xlabel: str, ylabel: str, title: str):
    """
    plot scatter plot
    :param xlabel:
    :param ylabel:
    :param x:
    :param y:
    :return:
    """

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 6), facecolor='w', edgecolor='k', dpi=200)  # figsize=(w,h)
    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, hspace=0.4, top=0.8, wspace=0.05)

    plt.scatter(x, y, marker='^', c='b', s=50, edgecolors='blue', alpha=0.8, label=ylabel)

    plt.title(title)

    # plt.legend(loc="upper right", markerscale=1, fontsize=16)

    ax.set_xlabel(xlabel, fontsize=18)
    ax.set_ylabel(ylabel, fontsize=18)
    ax.tick_params(labelsize=16)

    plt.grid(True)

    plt.show()


def plot_altitude_bias_by_month(df: pd.DataFrame, model_column: str, obs_column: str,
                                altitude_column: str, cbar_label: str,
                                fig_title: str, bias=False):
    """
    plot station locations and their values
    :param altitude:
    :param model:
    :param obs:
    :param bias:
    :param fig_title:
    :return: map show
    """
    import matplotlib.pyplot as plt
    import matplotlib as mpl

    # data:
    df['bias'] = df[model_column] - df[obs_column]

    months = [11, 12, 1, 2, 3, 4]
    month_names = ['Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr']
    nrows = len(months)

    fig = plt.figure(figsize=(5, 24), dpi=200)
    # fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, hspace=1.5, top=0.95, wspace=0.05)

    # plot in each month:
    for m in range(len(months)):
        ax = plt.subplot(len(months), 1, m + 1)

        # data:
        monthly_data = df[df.index.month == months[m]]
        station_group = monthly_data.groupby('station_id')
        station_mean_bias = station_group[['bias']].mean().values[:, 0]
        station_mean_height = station_group[['altitude']].mean().values[:, 0]

        lon = df['longitude']
        lat = df['latitude']

        # ----------------------------- cbar -----------------------------
        if np.max(station_mean_bias) - np.min(station_mean_bias) < 10:
            round_number = 2
        else:
            round_number = 0

        n_cbar = 10
        vmin = round(np.min(station_mean_bias) / n_cbar, round_number) * n_cbar
        vmax = round(np.max(station_mean_bias) / n_cbar, round_number) * n_cbar

        if bias:
            cmap = plt.cm.coolwarm
            vmin = max(np.abs(vmin), np.abs(vmax)) * (- 1)
            vmax = max(np.abs(vmin), np.abs(vmax))
        else:
            cmap = plt.cm.YlOrRd

        vmin = -340
        vmax = 340

        bounds = np.linspace(vmin, vmax, n_cbar + 1)
        norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

        sc = plt.scatter(lon, lat, c=station_mean_bias, edgecolor='black',
                         # transform=ccrs.PlateCarree(),
                         zorder=2, norm=norm, vmin=vmin, vmax=vmax, s=50, cmap=cmap)

        # ----------------------------- color bar -----------------------------
        cb = plt.colorbar(sc, orientation='horizontal', shrink=0.7, pad=0.1, label=cbar_label)
        cb.ax.tick_params(labelsize=10)

        # ax.xaxis.set_ticks_position('top')

        ax.gridlines(draw_labels=False)

        ax.text(0.93, 0.95, month_names[m],
                horizontalalignment='right', verticalalignment='top', transform=ax.transAxes)

    plt.tight_layout(pad=0.1, w_pad=0.3, h_pad=0.6)
    plt.suptitle(f'monthly daytime (8h 17h) \n mean bias at MeteoFrance stations')

    plt.show()
    print(f'got plot')


def plot_scatter_contourf(lon: xr.DataArray, lat: xr.DataArray, cloud: xr.DataArray, cbar_label: str,
                          lon_mf: np.ndarray, lat_mf: np.ndarray, value: np.ndarray, cbar_mf: str,
                          bias=False, bias_mf=True):

    """
    to plot meteofrance stational value and a colorfilled map.

    :param cbar_mf:
    :param lat_mf:
    :param lon_mf:
    :param cloud:
    :param bias:
    :param cbar_label: label of color bar
    :param lon:
    :param lat:
    :return: map show
    """

    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    import datetime as dt

    hours = [x for x in range(8, 18, 1)]
    dates = ['2004-11-01', '2004-12-01', '2005-01-01', '2005-02-01', '2005-03-01', '2005-04-01']
    month_names = ['Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr']

    fig = plt.figure(figsize=(10, 20), dpi=300)
    # fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, hspace=1.5, top=0.95, wspace=0.05)

    # plot in each hour:
    for h in range(len(hours)):

        ax = plt.subplot(len(hours)/2, 2, h + 1, projection=ccrs.PlateCarree())

        print(f'plot hour = {hours[h]:g}')

        # ----------------------------- mean cloud fraction -----------------------------
        # data:

        hourly_cloud = cloud.sel(time=dt.time(hours[h])).mean(axis=0)

        # set map
        reu = get_lonlatbox('reu')
        ax.set_extent(reu, crs=ccrs.PlateCarree())
        # lon_left, lon_right, lat_north, lat_north

        ax.coastlines('50m')
        ax.add_feature(cfeature.LAND.with_scale('10m'))

        # Plot Colorfill of hourly mean cloud fraction
        # normalize color to not have too dark of green at the top end
        clevs = np.arange(60, 102, 2)
        cf = ax.contourf(lon, lat, hourly_cloud, clevs, cmap=plt.cm.Greens,
                         norm=plt.Normalize(60, 102), transform=ccrs.PlateCarree())

        # cb = plt.colorbar(cf, orientation='horizontal', pad=0.1, aspect=50)
        cb = plt.colorbar(cf, orientation='horizontal', shrink=0.7, pad=0.05, label=cbar_label)
        # cb.set_label(cbar_label)

        # ----------------------------- hourly mean bias wrf4.1 - mf -----------------------------

        # data:

        hourly_bias = value[value.index.hour==hours[h]]
        hourly_bias = hourly_bias.groupby('station_id').mean().values.reshape((37,))

        vmax = 240
        vmin = vmax * -1

        if bias_mf:
            cmap = plt.cm.coolwarm
            vmin = max(np.abs(vmin), np.abs(vmax)) * (- 1)
            vmax = max(np.abs(vmin), np.abs(vmax))
        else:
            cmap = plt.cm.YlOrRd

        n_cbar = 20

        bounds = np.linspace(vmin, vmax, n_cbar + 1)
        norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
        # ----------------------------------------------------------
        sc = plt.scatter(lon_mf, lat_mf, c=hourly_bias, edgecolor='black',
                         # transform=ccrs.PlateCarree(),
                         zorder=2, norm=norm, vmin=vmin, vmax=vmax, s=50, cmap=cmap)

        # ----------------------------- color bar -----------------------------
        cb = plt.colorbar(sc, orientation='vertical', shrink=0.7, pad=0.05, label=cbar_mf)
        cb.ax.tick_params(labelsize=10)

        # ----------------------------- end of plot -----------------------------
        ax.xaxis.set_ticks_position('top')

        ax.gridlines(draw_labels=False)

        ax.text(0.98, 0.95, f'{hours[h]:g}h00\nDJF mean\n2004-2005', horizontalalignment='right', verticalalignment='top',
                transform=ax.transAxes)

        ax.text(0.01, 0.16, f'0.05x0.05 degree\nMVIRI/SEVIRI on METEOSAT',
                horizontalalignment='left', verticalalignment='top', transform=ax.transAxes)
    # plt.tight_layout(pad=0.1, w_pad=0.3, h_pad=0.6)
    plt.suptitle(f'cloud fraction at daytime hours \n as mean of DJF during 2004 - 2005')

    plt.show()
    print(f'got plot')


def get_lonlatbox(area:str):
    """
    get lonlat box of an area by names
    :param area:
    :return: list
    """

    if area == 'reu':
        box = [55, 56, -21.5, -20.8]

    if area == 'swio':
        box = [20, 110, -50, 9]

    if area == 'reu_mau':

        box = [52, 60, -17.9, -23]

    return box


def cluster_mean_gaussian_mixture(var_history, n_components, max_iter, cov_type):
    """
    input days with similar temp profile, return a dataframe with values = most common cluster mean.

    :param day: nd.array(datatime.date)
    :param var_history: pd.DateFrame
    :param var_similar: pd.DateFrame
    :param n_components:
    :param max_iter:
    :param cov_type:
    :return: pd.DateFrame of DateTimeIndex
    """

    from collections import Counter
    from sklearn.mixture import GaussianMixture
    from sklearn.neural_network import MLPRegressor

    # clustering by Gaussian Mixture
    gm = GaussianMixture(n_components=n_components, max_iter=max_iter, covariance_type=cov_type)

    var_clusters = gm.fit(var_history)

    cluster_mean = var_clusters.means_

    labels = gm.predict(var_history)

    return cluster_mean, labels


def plot_daily_cluster_mean(mean, locations, labels, ylabel, title):

    fig = plt.figure(figsize=(10, 6), dpi=220)
    # fig.suptitle(fig_title)

    ax = fig.add_subplot(1, 2, 1)
    ax.set_aspect(aspect=0.015)
    # ----------------------------- plotting -----------------------------

    colors = ['blue', 'red', 'orange']
    markers = ['o', '^', 's']
    group_names = ['group 1', 'group 2', 'group 3']

    # get x in hours, even when only have sunny hours:
    x = range(8, 18)

    for c in range(mean.shape[0]):
        plt.plot(x, mean[c, :], color=colors[c], marker=markers[c], label=group_names[c])

    plt.hlines(0,8,17,colors='black')

    # plt.text(0.98, 0.95, 'text',
    #          horizontalalignment='right', verticalalignment='top', transform=ax.transAxes)

    plt.legend(loc='upper right', prop={'size': 8})
    plt.title(title)
    # ----------------------------- location of group members -----------------------------

    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import matplotlib as mpl

    ax = fig.add_subplot(1, 2, 2, projection=ccrs.PlateCarree())
    ax.set_extent([55, 56, -21.5, -20.8], crs=ccrs.PlateCarree())
    ax.coastlines('50m')

    # ----------------------------- plot cloud fraction from CM SAF -----------------------------

    # cloud fraction cover
    cfc_cmsaf = f'{DIR:s}/local_data/obs/CFC.cmsaf.hour.reu.DJF.nc'
    cloud = xr.open_dataset(cfc_cmsaf).CFC

    mean_cloud = cloud.mean(dim='time')

    clevs = np.arange(60, 82, 2)
    cf = ax.contourf(cloud.lon, cloud.lat, mean_cloud, clevs, cmap=plt.cm.Greens,
                     norm=plt.Normalize(60, 82), transform=ccrs.PlateCarree(), zorder=1)

    # cb = plt.colorbar(cf, orientation='horizontal', pad=0.1, aspect=50)
    cb = plt.colorbar(cf, orientation='horizontal', shrink=0.7, pad=0.05, label='daily mean cloud fraction CM SAF')


    # ----------------------------- locations -----------------------------
    # plot location of stations
    for i in range(len(locations)):
        label = labels[i]

        lon = locations['longitude'].values[i]
        lat = locations['latitude'].values[i]

        plt.scatter(lon, lat, color=colors[label],
                    edgecolor='black', zorder=2, s=50, label=group_names[label] if i == 0 else "")

    # sc = plt.scatter(locations['longitude'], locations['latitude'], c=labels,
    #                  edgecolor='black', zorder=2, s=50)


    ax.gridlines(draw_labels=True)

    plt.xlabel(u'$hour$')
    plt.ylabel(ylabel)
    plt.legend(loc='upper right', prop={'size': 8})

    print("waiting for the plot\n")
    plt.show()

