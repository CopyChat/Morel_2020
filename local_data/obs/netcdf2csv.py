#!/Users/ctang/anaconda3/bin/python3.7
"""
========
Ctang, this code read 1 pixel of netcdf file,
       and then covert to csv in local time.

======== 
"""
import sys
import click
import netCDF4
import numpy as np
import pandas as pd
from dateutil import tz


def utc2local(test, df):
    """
    convert utc time to local time
    :param test:
    :param df:
    :return: pd.DataFrame
    """
    time_local = []
    for i in range(df.__len__()):
        utc = df.index[i].replace(tzinfo=tz.tzutc())
        local_time = utc.astimezone(tz.tzlocal())
        local_time = local_time.replace(tzinfo=None)
        time_local.append(local_time)

        if test:
            print(utc, local_time)

    df['local_time'] = time_local

    out_df = pd.DataFrame(data=df.values, columns=df.columns)

    out_df['local_time'] = pd.to_datetime(out_df['local_time'], format='%Y-%m-%d %H:%M:%S')
    out_df = out_df.set_index('local_time')

    return out_df


def read_netcdf(test: bool, netcdf: str, var: str, lon_loc: int, lat_loc: int):
    """
    to read input netcdf file return a pd.DataFrame
    :param lat_loc: location index of moufia
    :param lon_loc:
    :param test:
    :param netcdf:
    :param var:
    :return: pd.DataFrame
    """
    if test:
        print('start to read')

    nc = netCDF4.Dataset(netcdf)

    times = nc.variables['time']
    jd: np.ndarray = netCDF4.num2date(times[:], times.units)

    v = nc.variables[var][:, lon_loc, lat_loc]
    if test:
        print(f'starting to read var, it will takes a while...')
    # hs = pd.Series(h, index=jd)
    df = pd.DataFrame(np.array(v), index=jd, columns=[var])

    if test:
        print(f'done, reading var in dimension: {str(v.shape):s}')

    return df


@click.command()
@click.argument('netcdf', nargs=1)
@click.argument('var', nargs=1)
@click.option('--test', '-t', is_flag=True, default=False, help='output testing info ?')
@click.option('--output_prefix', '-prefix', help='output prefix str')
@click.option('--lon_loc', '-lon', help='location of lon')
@click.option('--lat_loc', '-lat', help='location of lat')
def main(test, netcdf, var, output_prefix, lon_loc, lat_loc):
    """
    read netcdf var one point, and output csv
    :param output_prefix:
    :param lat_loc:
    :param lon_loc:
    :param test:
    :param netcdf:
    :param var:
    :return:
    """
    # read netcdf
    df_utc = read_netcdf(test, netcdf, var, lon_loc=lon_loc, lat_loc=lat_loc)

    # change time zone:
    df_local = utc2local(test, df_utc)

    # output print:
    print('DataTime,' + str(var))
    for i in range(len(df_local)):
        print(f'{output_prefix:s}{str(df_local.index[i]):s},{df_local[var][i]:0.4f}')


if __name__ == "__main__":
    sys.exit(main())
