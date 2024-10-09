#!/bin/bash
#SBATCH --partition intelsr_long
#SBATCH --time 7-00:00:00
#SBATCH --job-name abc_old_sumstats
#SBATCH --output log/log_abc_old_sumstats.%j.txt
#SBATCH --ntasks=6
#SBATCH --nodes=6
#SBATCH --cpus-per-task=32
#SBATCH --depend=afterany:17076448
#SBATCH --account ag_irumls_mircea
#SBATCH --mail-type=END
#SBATCH --mail-user=jonas.arruda@uni-bonn.de

# Define variables
PARTITION="intelsr_long"
TIME="7-00:00:00"
NODES=6
CPUS=32
PORT=7412
JOBNAME="abc_old_sumstats"

# Source environment
source ~/CellMigration/env.sh

# Disable tqdm progress bars
TQDM_DISABLE=1
export TQDM_DISABLE

#python3.10 ~/CellMigration/synth_data_params_bayesflow/cell_migration_bayesflow.py --train
#python3.10 ~/CellMigration/synth_data_params_bayesflow/cell_migration_pyabc.py

# Submit the job using the same parameters
./submit_job_L.sh $PORT $NODES $PARTITION $TIME $CPUS $JOBNAME ~/CellMigration/synth_data_params_bayesflow/cell_migration_pyabc.py
