#!/bin/bash - 
#======================================================
#
#          FILE: select_at_each_station.sh
# 
USAGE="./select_at_each_station.sh"
# 
#   DESCRIPTION: select data from sarah-e & era5_land hourly data
#                to the stations of MeteoFrance over la reunion
# 
#       OPTIONS: ---
#  REQUIREMENTS: ---
#          BUGS: --- unknown
#         NOTES: ---
#        AUTHOR: |CHAO.TANG| , |chao.tang.1@gmail.com|
    #  ORGANIZATION: 
#       CREATED: 09/24/2019 15:08
#      REVISION: 1.0
#=====================================================
set -o nounset           # Treat unset variables as an error
. ~/Shell/functions.sh   # ctang's functions

while getopts ":tf:" opt; do
    case $opt in
        t) TEST=1 ;;
        f) file=$OPTARG;;
        \?) echo $USAGE && exit 1
    esac
done
shift $(($OPTIND - 1))
#=================================================== 

DIR=/Users/ctang/Microsoft_OneDrive/OneDrive/CODE/Morel_2020/local_data/obs

# ----------------------------- data -----------------------------
# grided data in reso=0.05
sarah_e=$DIR/SIS.hour.20041015-20040430.sarah-e.reu.nc
var=SIS

# --------------------- station meteofrance ----------------------
station_file=$DIR/station_meteoFrance.csv # output of DataBase

output=$DIR/sarah_e.20041015-20050430.station_meteoFrance.csv

# ----------------------------- get info -----------------------------
num_line=$(wc -l $station_file | awk '{print $1}')

num_sta=$(wc -l $station_file | awk '{print $1-1}')

for line in $(seq -s " " 2 $num_line)
do
    station_id=$(awk -F "," 'NR=='$line'{print $1}' $station_file)
    longitude=$(awk -F "," 'NR=='$line'{print $7}' $station_file)
    latitude=$(awk -F "," 'NR=='$line'{print $6}' $station_file)
    name=$(awk -F "," 'NR=='$line'{print $2}' $station_file)

    echo $line $station_id $latitude $longitude

    # ----------------------------- select grid points

    reso=0.025

    #lat_left=$[ latitude + reso ]

    lat_left=$(echo "" | awk '{print '$latitude'-'$reso'}')
    lat_right=$(echo "" | awk '{print '$latitude'+'$reso'}')
    lon_left=$(echo "" | awk '{print '$longitude'-'$reso'}')
    lon_right=$(echo "" | awk '{print '$longitude'+'$reso'}')

    echo $lat_left,$lat_right,$lon_left,$lon_right
    cdo sellonlatbox,$lon_left,$lon_right,$lat_left,$lat_right $sarah_e \
        $station_id.nc.temp

    netcdf2csv.py $station_id.nc.temp $var -prefix $station_id, -lon 0 -lat 0 > $station_id.csv.temp

    awk 'NR>1' $station_id.csv.temp >> $output

done





