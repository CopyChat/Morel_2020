#!/Users/ctang/anaconda3/bin/python3.7
"""
========
Ctang, this code read 1 pixel of netcdf file,
       and then covert to csv in local time.
 to call this code :
 netcdf2csv.2.py SWDOWN.hour.d03.wrf3.5.nc SWDOWN -time Times -lon 55.5 -lat -21.2 -prefix jjj,
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


def read_netcdf(test: bool, netcdf: str, var: str, time_var: str, start_time: str, lon_loc: int, lat_loc: int):
    """
    to read input netcdf file return a pd.DataFrame
    :param start_time: to add the list of timestamp to the output pd.DataFrame
    :param time_var: name of time var
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

    times = nc.variables[time_var]
    # jd: np.ndarray = netCDF4.num2date(times[:], times.units)
    jd = np.array(times)

    # find index:
    index_lon = find_lonlat_index(start=55.02496, end=56.04868, number=140, input_location=lon_loc, reso=0.05)
    index_lat = find_lonlat_index(start=-21.59586, end=-20.64304, number=140, input_location=lat_loc, reso=0.05)

    v = nc.variables[var][:, index_lon, index_lat]

    dt_list = get_dt_list(start=start_time, number=v.shape[0])

    if test:
        print(f'starting to read var, it will takes a while...')
    df = pd.DataFrame(np.array(v), index=dt_list, columns=[var])

    if test:
        print(f'done, reading var in dimension: {str(v.shape):s}')

    return df


def find_lonlat_index(start, end, number, input_location, reso):
    """
    to find lon/lat index according to the input of lon/lat
    :param reso: resolution of grided data
    :param start:
    :param end:
    :param number:
    :param input_location:
    :return:
    """
    array_all = [start + i * (end - start)/number for i in range(1, number + 1)]

    for j in range(1, number + 1):
        if float(input_location) - 0.5 * reso <= array_all[j] <= float(input_location) + 0.5 * reso:
            index = j
            break

    return index


def get_dt_list(start, number):

    import datetime
    start = datetime.datetime.strptime(start, "%Y-%m-%d %H:%M:%S")
    end = datetime.datetime.strptime("2005-04-30 18:00:00", "%Y-%m-%d %H:%M:%S")
    date_generated = [start + datetime.timedelta(hours=x) for x in range(0, number)]

    return date_generated


@click.command()
@click.argument('netcdf', nargs=1)
@click.argument('var', nargs=1)
@click.option('--test', '-t', is_flag=True, default=False, help='output testing info ?')
@click.option('--output_prefix', '-prefix', help='output prefix str')
@click.option('--time_var', '-time', help='time var name')
@click.option('--longitude', '-lon', help='longitude')
@click.option('--latitude', '-lat', help='latitude')
def main(test, netcdf, var, output_prefix, time_var, longitude, latitude):
    """
    read netcdf var one point, and output csv
    :param time_var:
    :param longitude:
    :param latitude:
    :param output_prefix:
    :param test:
    :param netcdf:
    :param var:
    :return:
    """
    # read netcdf
    df_utc = read_netcdf(test, netcdf, var, start_time="2004-10-15 00:00:00",
                         time_var=time_var, lon_loc=longitude, lat_loc=latitude)

    if test:
        print(f'read done')
    # change time zone:
    df_local = utc2local(test, df_utc)

    # get dt list:

    # output print:
    if test:
        print('DataTime,' + str(var))
    for i in range(len(df_local)):
        print(f'{output_prefix:s}{str(df_local.index[i]):s},{np.array(df_local[var])[i]:0.4f}')


if __name__ == "__main__":
    sys.exit(main())
