#!/bin/bash
#SBATCH --partition intelsr_medium
#SBATCH --time 1-00:00:00
#SBATCH --job-name sim_cell_migration
#SBATCH --output log/log_sim_cell_migration.%j.txt
#SBATCH --cpus-per-task 1
#SBATCH --array=0-999%192
#SBATCH --depend=afterany:17076159
#SBATCH --mail-type=END
#SBATCH --mail-user=jonas.arruda@uni-bonn.de

source ~/CellMigration/env.sh

TQDM_DISABLE=1
export TQDM_DISABLE

python3.10 ~/CellMigration/synth_data_params_bayesflow/cell_migration_bayesflow.py --presimulate
