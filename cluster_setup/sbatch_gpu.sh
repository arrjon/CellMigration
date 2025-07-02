#!/bin/bash
#SBATCH --partition mlgpu_medium
#SBATCH --time 1-00:00:00
#SBATCH --job-name train_cell_migration
#SBATCH --output log/log_train_cell_migration.%j.txt
#SBATCH --cpus-per-task 1
#SBATCH --mem-per-cpu 64GB
#SBATCH --gpus 1
#SBATCH --array=0-2
#SBATCH --depend=afterany:22392705
#SBATCH --mail-type=END
#SBATCH --mail-user=jonas.arruda@uni-bonn.de

source ~/CellMigration/env.sh

TQDM_DISABLE=1
export TQDM_DISABLE

python3.10 ~/CellMigration/synth_data_params_bayesflow/cell_migration_bayesflow.py --train
#python3.10 ~/CellMigration/synth_data_params_bayesflow/cell_migration_nn.py --train
