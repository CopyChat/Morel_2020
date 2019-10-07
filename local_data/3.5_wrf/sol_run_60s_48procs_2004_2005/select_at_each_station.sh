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

DIR=/Users/ctang/Microsoft_OneDrive/OneDrive/CODE/Morel_2020/local_data/3.5_wrf/sol_run_60s_48procs_2004_2005

# ----------------------------- data -----------------------------
grid_nc=$DIR/SWDOWN.hour.d03.wrf3.5.nc
var=SWDOWN
reso=0.0073
# --------------------- station meteofrance ----------------------
station_file=$DIR/station_meteofrance.2004-2005.with.rg.csv # output of DataBase

output=$DIR/wrf35.20041015-20050430.station_meteoFrance.csv

# ----------------------------- get info -----------------------------
num_line=$(wc -l $station_file | awk '{print $1}')

num_sta=$(wc -l $station_file | awk '{print $1-1}')

for line in $(seq -s " " 2 $num_line)
do
    sta_no=$(echo "" | awk '{print '$line'-1}')

    station_id=$(awk -F "," 'NR=='$line'{print $1}' $station_file)
    longitude=$(awk -F "," 'NR=='$line'{print $4}' $station_file)
    latitude=$(awk -F "," 'NR=='$line'{print $3}' $station_file)
    name=$(awk -F "," 'NR=='$line'{print $2}' $station_file)

    echo $sta_no $station_id $latitude $longitude

    # ----------------------------- select grid points

    #lat_left=$[ latitude + reso ]

    #lat_left=$(echo "" | awk '{print '$latitude'-'$reso'*0.5}')
    #lat_right=$(echo "" | awk '{print '$latitude'+'$reso'*0.5}')
    #lon_left=$(echo "" | awk '{print '$longitude'-'$reso'*0.5}')
    #lon_right=$(echo "" | awk '{print '$longitude'+'$reso'*0.5}')

    #echo $lat_left,$lat_right,$lon_left,$lon_right
    #cdo sellonlatbox,$lon_left,$lon_right,$lat_left,$lat_right $grid_nc \
        #$station_id.nc.temp

    ./netcdf2csv.2.py SWDOWN.hour.d03.wrf3.5.nc SWDOWN -time Times -lon $longitude -lat $latitude -prefix $station_id, > $station_id.csv.temp

    cat $station_id.csv.temp >> $output

done





