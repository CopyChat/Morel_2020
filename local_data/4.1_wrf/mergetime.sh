#u!/bin/bash - 
#======================================================
#
#          FILE: mergetime.sh
# 
USAGE="./mergetime.sh"
# 
#   DESCRIPTION: 
# 
#       OPTIONS: ---
#  REQUIREMENTS: ---
#          BUGS: --- unknown
#         NOTES: ---
#        AUTHOR: |CHAO.TANG| , |chao.tang.1@gmail.com|
#  ORGANIZATION: 
#       CREATED: 11/13/2019 06:08
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
# ----------------------------------------------------------

netcdf_list=$(ls addout_d03_200?-??-??_00:00:00.nc)

echo $netcdf_list

for file in ${netcdf_list}
do
    #cdo sinfo $file
    echo $file
done

# first, set the env of skip:
export SKIP_SAME_TIME=1

# then, mergetime:

cdo mergetime addout_d03_200?-??-??_??:00:00.nc output.nc

# before done, check if number of time is right

cdo sinfo output.nc


# for select timesteps, could use the following command:
# ncks -d Time,0,433 in.nc out.nc
