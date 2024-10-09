#!/bin/sh

export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1

cur_PWD=${1}
IP=${2}
PORT=${3}
TIME=${4}
CPUSPERTASK=${5}

echo "cur_PWD: $cur_PWD"
echo "IP: $IP"
echo "PORT: $PORT"
echo "TIME: $TIME"
echo "TIME: ${TIME:0:1}d"

cd ${cur_PWD}

# Source module
source ~/CellMigration/env.sh

# Start redis-worker
cd /home/jarruda_hpc/CellMigration/cellMigration/bin/
abc-redis-worker  --host=${IP} --port ${PORT} --runtime ${TIME:0:1}d --processes ${CPUSPERTASK} --daemon false




