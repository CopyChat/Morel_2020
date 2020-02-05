"""
functions used for get and plot geo data
"""

__version__ = f'Version 1.0  \nTime-stamp: <2019-02-21>'
__author__ = "ChaoTANG@univ-reunion.fr"

import pandas as pd
import numpy as np


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


def query_data(mysql_query: str):
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

    df = df.set_index('DateTime')
    return df


def plot_station_value(lon: pd.DataFrame, lat: pd.DataFrame, value: np.array, cbar_label: str,
                       fig_title: str):
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
    cmap = plt.cm.YlOrRd

    if np.max(value) - np.min(value) < 10:
        round_number = 2
    else:
        round_number = 0

    n_cbar = 10
    vmin = round(np.min(value) / n_cbar, round_number) * n_cbar
    vmax = round(np.max(value) / n_cbar, round_number) * n_cbar

    bounds = np.linspace(vmin, vmax, n_cbar + 1)
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    # ----------------------------------------------------------
    sc = plt.scatter(lon, lat, c=value, edgecolor='black',
                     # transform=ccrs.PlateCarree(),
                     zorder=2, norm=norm, vmin=vmin, vmax=vmax, s=50, cmap=cmap)

    # ----------------------------- color bar -----------------------------
    cb = plt.colorbar(sc, orientation='horizontal', shrink=0.8, pad=0.05, label=cbar_label)
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


def plot_hourly_curve_by_month(df: pd.DataFrame, columns: list):
    """
    plot hourly curves by /month/ for the columns in list
    :param df:
    :param columns:
    :return:
    """

    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    # ----------------------------- set parameters -----------------------------
    months = [11, 12, 1, 2, 3, 4]
    colors = ['black', 'green', 'red']
    data_sources = ['MeteoFrance', 'SARAH-E', 'WRF3.5']

    # ----------------------------- set fig -----------------------------
    nrows = len(months)

    fig, axs = plt.subplots(nrows=nrows, ncols=1, figsize=(16, 9),
                            facecolor='w', edgecolor='k', dpi=200)  # figsize=(w,h)
    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, hspace=0.4, top=0.96, wspace=0.05)

    axs = axs.ravel()
    # ----------------------------- plotting -----------------------------

    from matplotlib.dates import DateFormatter
    for v in range(len(columns)):
            ax = plt.sca(axs[i])  # active this subplot

            month = months[i]
            data_slice = df[df.index.month == month]
            x = range(len(data_slice))
            plt.plot(x, data_slice[[columns[v]]], color=colors[v], label=data_sources[v])

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
            plt.title(data_slice.index[0].month_name())

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
        nrow = 2
        ncol = 3
    if by is None:
        nrow = 1
        ncol = 1

    n_plot = nrow * ncol

    fig, axs = plt.subplots(nrows=nrow, ncols=ncol,
                            figsize=(16, 10), facecolor='w', edgecolor='k', dpi=200)  # figsize=(w,h)

    if by == 'Month':
        axs = axs.ravel()

    months = [11, 12, 1, 2, 3, 4]
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
            i == max(range(len(months)))

        all_var = pd.DataFrame()
        for col in range(len(columns)):
            # print(f'for var = {columns[col]:s}')
            # calculate normalised value:
            var = pd.DataFrame()
            var['target'] = data_slice[columns[col]]
            var['Hour'] = data_slice.index.hour
            var['var'] = [columns[col] for x in range(len(data_slice))]
            all_var = all_var.append(var)

        seaborn.boxplot(x='Hour', y='target', hue='var', data=all_var, ax=ax)

        ax.set_xlim(5, 20)
        # ax.set_ylim(0, 1.1)
        if by is not None:
            ax.set_title(f'{by:s} = {months[i]:g}')

    # Get the handles and labels. For this example it'll be 2 tuples
    # of length 4 each.
    handles, labels = ax.get_legend_handles_labels()

    # When creating the legend, only use the first two elements
    # to effectively remove the last two.
    plt.legend(handles[0:3], labels[0:3], bbox_to_anchor=(0.95, 0.95), borderaxespad=0.,
               loc="upper right", fontsize=18)

    plt.ylabel(f'distribution')
    plt.title(f'SSR distribution between 5AM - 8PM', fontsize=18)
    ax.set_xlabel(f'Hour', fontsize=18)
    ax.set_ylabel(r'$SSR\ (W/m^2)$', fontsize=18)
    plt.title(data_slice.index[0].month_name())

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
        nrow = 2
        ncol = 3
    if by is None:
        nrow = 1
        ncol = 1

    n_plot = nrow * ncol

    fig, axs = plt.subplots(nrows=nrow, ncols=ncol,
                            figsize=(16, 10), facecolor='w', edgecolor='k', dpi=200)  # figsize=(w,h)

    if by == 'Month':
        axs = axs.ravel()

    months = [11, 12, 1, 2, 3, 4]
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

        seaborn.boxplot(x='Hour', y='target', hue='var', data=all_var, ax=ax)

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
        plt.legend(handles[0:3], labels[0:3], bbox_to_anchor=(0.95, 0.95), borderaxespad=0.,
                   loc="5pper right", fontsize=18)

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
    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, hspace=0.4, top=0.8, wspace=0.05)

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


