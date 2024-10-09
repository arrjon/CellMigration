#!/bin/sh

cur_PWD=${1}
IP=${2}
PORT=${3}
JOBNAME=${4}
PYHTONFILE=${5}

cd ${cur_PWD}

echo "cur_PWD: $cur_PWD"
echo "IP: $IP"
echo "PORT: $PORT"
echo "PYHTONFILE: $PYHTONFILE"

# Source module
source ~/CellMigration/env.sh

# Start the python script
python ${PYHTONFILE} --port ${PORT} --ip ${IP} > log/out_${JOBNAME}.txt 2> log/err_${PORT}_${JOBNAME}.txt
