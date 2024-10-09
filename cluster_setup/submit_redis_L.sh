#!/bin/sh

cur_PWD=${1}
PORT=${2}

echo '#### cur_PWD= ####'${cur_PWD}
echo '#### PORT= ####'${PORT}

cd ${cur_PWD}

echo '#### This is the redis script job ####'

# Source module
source ~/CellMigration/env.sh

# Start redis-worker
/home/jarruda_hpc/CellMigration/redis-stable_new/src/./redis-server --port ${PORT} --protected-mode no
