This folder contains the last generation of the fitting result

synth_data_params_4_wass: The model with Wasserstein instead of the Euclidean distance
synth_data_params_4_wass_var: similar to the above model, but using the variance of the summary statistics instead of the mean
synth_data_params_4_wass_var_50cell: similar to the above model, but having a population of 50 cells

######################################################

Each folder will contain the following files:

### FOLDERS ###

db: the folder contains the database in sqlite3 format
log: usually empty, but was created in case one wants to enable the logging feature from pyABC
outplot: usually empty, unless one created a plot at the end of the fitting
testoutdir: usually empty, used only if one wants to make a test or debugging run to check the model output instead of saving them on the /tmp folder

### PYTHON SCRIPTS ###

Cell_migration.py: the main Python script that contains the Fitmulticell script to fit the model
cell_migration_test.py: a test script for testing purposes

### MODEL FILES ###

cell_movement_v21.xml: the model in MorpheusML format
Cell_migration_grid_v3_final2_invers: a tiff image that is required to construct the model spatial lattice

### OUTPUT FILES ###

err_xxx.txt: error file copied from the terminal of the main node during the run time
slurm-xxxx.out: output file copied from the terminal of the main node during the run time
redis_output.txt: output file copied from the terminal of the redis node during the run time
python_output.txt: output file copied from the terminal of the python running node during the run time
worker_1_output.txt: output file copied from the terminal of the worker node during the run time

### BATCH FILES ###

load_module.sh: one can write all required modules for the job and load them when submitting the job
kill_all.sh: batch file used to kill all running scripts on the cluster
submit_job_L.sh: The main batch script that needs to be submitted to the cluster which internally calls other needed scripts, eg., submit_python_L.sh
submit_python_L.sh: The batch script that runs the main Python code
submit_redis_L.sh: The batch script that runs and establishes the Redis server node
submit_worker_L.sh: The batch script that runs the model simulation

### miscellaneous ###

master_ip: a file that will be created on the fly to link the worker node to the main node


