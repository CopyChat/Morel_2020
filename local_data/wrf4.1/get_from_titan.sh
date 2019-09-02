#!/bin/bash - 
#======================================================
#
#          FILE: get_from_titan.sh
# 
USAGE="./get_from_titan.sh"
# 
#   DESCRIPTION: 
# 
#       OPTIONS: ---
#  REQUIREMENTS: ---
#          BUGS: --- unknown
#         NOTES: ---
#        AUTHOR: |CHAO.TANG| , |chao.tang.1@gmail.com|
    #  ORGANIZATION: 
#       CREATED: 06/20/19 15:35
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

rsync -acrxSPH ctang@10.82.80.222:/gpfs/labos/le2p/ctang/Morel_2020/wrf_ssr/run/namelist.* ./

rsync -acrxSPH ctang@10.82.80.222:/gpfs/labos/le2p/ctang/Morel_2020/wrf_ssr/run/myout*txt ./

rsync -acrxSPH ctang@10.82.80.222:/gpfs/labos/le2p/ctang/Morel_2020/wrf_ssr/run/rsl.*.* ./

rsync -acrxSPH ctang@10.82.80.222:/gpfs/labos/le2p/ctang/Morel_2020/wrf_ssr/run/myout*txt ./




