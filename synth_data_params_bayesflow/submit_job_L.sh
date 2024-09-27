#!/bin/sh
# Master script
#BATCH --tasks-per-node=1

PORT=${1}
N_NODES=${2}
PARTITION=${3}
TIME=${4}
CPUSPERTASK=${5}
PYTHONFILE=${6}

echo "PORT: $PORT"
echo "N_NODES: $N_NODES"
echo "PARTITION: $PARTITION"
echo "TIME: $TIME"
echo "CPUSPERTASK: $CPUSPERTASK"
echo "PYTHONFILE: $PYTHONFILE"

if [ -z "${N_NODES}" ]
then
        N_NODES=1
fi

# Soruce Modules
source ~/CellMigration/env.sh


# Submit redis 
srun --nodes=1 --ntasks=1 --cpus-per-task=${CPUSPERTASK}  bash -c 'hostname > master_ip && ${PWD}/submit_redis_L.sh $0 $1' ${PWD} ${PORT} > log/redis_output.txt &
# Wait for redis to start
sleep 10

# Retrieve IP of compute node
IP_long=`cat master_ip | tr '\n' ' '`
IP="${IP_long%%.*}"
MASTER_IP=`getent hosts ${IP}i | cut -d' ' -f1`
REDIS_IP=$(host ${IP_long} | awk '{ print $4 }')
echo 'Total number of requested nodes = '$((${N_NODES}))
echo 'ip= '${IP}
echo 'masterip= ' ${MASTER_IP}
echo 'REDIS_IP= ' ${REDIS_IP}

# Start redis-worker
for i in $(seq 1 $((N_NODES-2)))
do
    srun --nodes=1 --ntasks=1 --cpus-per-task=${CPUSPERTASK} submit_worker_L.sh ${PWD} ${REDIS_IP} ${PORT} ${TIME} ${CPUSPERTASK} > log/worker_${i}_output.txt &

done

# Start python script
srun --nodes=1 --ntasks=1 --cpus-per-task=${CPUSPERTASK} submit_python_L.sh ${PWD} ${REDIS_IP} ${PORT} ${PYTHONFILE} > log/python_output.txt

