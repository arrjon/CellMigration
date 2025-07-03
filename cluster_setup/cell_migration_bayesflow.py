import os
import sys
import pickle
from functools import partial

import numpy as np
import pyabc
from fitmulticell import model as morpheus_model
from fitmulticell.sumstat import SummaryStatistics
from matplotlib import pyplot as plt

import tensorflow as tf
import bayesflow as bf
from bayesflow.simulation import GenerativeModel, Prior, Simulator

from load_bayesflow_model import load_model, custom_loader
from summary_stats import reduce_to_coordinates
from plotting_routines import plot_compare_summary_stats, plot_trajectory, plot_autocorrelation

# get the job array id and number of processors
run_args = sys.argv[1]
run_args = run_args.split('--')
job_array_id = int(os.environ.get('SLURM_ARRAY_TASK_ID', 0))
n_procs = int(os.environ.get('SLURM_CPUS_PER_TASK', 1))
print(job_array_id)

on_cluster = True
presimulate = False
training = False

if 'train' in run_args:
    training = True
if 'presimulate' in run_args:
    presimulate = True

# %%
if on_cluster:
    gp = '/home/jarruda_hpc/CellMigration/synth_data_params_bayesflow'
else:
    gp = os.getcwd()

# defining the mapping of parameter inside the model xml file. the dictionary name is for
# parameter name, and the value are the mapping values, to get the map value for parameter
# check here: https://fitmulticell.readthedocs.io/en/latest/example/minimal.html#Inference-problem-definition


par_map = {
    'gradient_strength': './CellTypes/CellType/Constant[@symbol="gradient_strength"]',
    'move.strength': './CellTypes/CellType/Constant[@symbol="move.strength"]',
    'move.duration.mean': './CellTypes/CellType/Constant[@symbol="move.duration.mean"]',
    'cell_nodes_real': './Global/Constant[@symbol="cell_nodes_real"]',
}

model_path = gp + "/cell_movement_v24.xml"  # time step is 30sec
# defining the summary statistics function
max_sequence_length = 120
min_sequence_length = 0
only_longest_traj_per_cell = True  # mainly to keep the data batchable
sumstat = SummaryStatistics(sum_stat_calculator=partial(reduce_to_coordinates,
                                                        minimal_length=min_sequence_length,
                                                        maximal_length=max_sequence_length,
                                                        only_longest_traj_per_cell=only_longest_traj_per_cell))

if on_cluster:
    # define the model object
    model = morpheus_model.MorpheusModel(
        model_path, par_map=par_map, par_scale="log10",
        show_stdout=False, show_stderr=False,
        executable="ulimit -s unlimited; /home/jarruda_hpc/CellMigration/morpheus-2.3.7",
        clean_simulation=True,
        raise_on_error=False, sumstat=sumstat)

    # todo: remember also change tiff path in model.xml!
else:
    # define the model object
    model = morpheus_model.MorpheusModel(
        model_path, par_map=par_map, par_scale="log10",
        show_stdout=False, show_stderr=False,
        clean_simulation=True,
        raise_on_error=False, sumstat=sumstat)

# parameter values used to generate the synthetic data
obs_pars = {
    'gradient_strength': 100.,  # strength of the gradient of chemotaxis
    'move.strength': 10.,  # strength of directed motion
    'move.duration.mean': 0.1,  # mean of exponential distribution (1/seconds)
    'cell_nodes_real': 50.,  # volume of the cell
}

# define parameters' limits
obs_pars_log = {key: np.log10(val) for key, val in obs_pars.items()}
limits = {'gradient_strength': (1, 10000), #(10 ** 4, 10 ** 8),
          'move.strength': (1, 100),
          'move.duration.mean': (1e-4, 30), #(math.log10((10 ** -2) * 30), math.log10((10 ** 4))), # smallest time step in simulation 5
          'cell_nodes_real': (1, 300)}
limits_log = {key: (np.log10(val[0]), np.log10(val[1])) for key, val in limits.items()}

prior = pyabc.Distribution(**{key: pyabc.RV("uniform", loc=lb, scale=ub-lb)
                              for key, (lb, ub) in limits_log.items()})
param_names = ['$m_{\\text{dir}}$', '$m_{\\text{rand}}$', '$w$', '$a$']
log_param_names = ['$\log_{10}(m_{\\text{dir}})$', '$\log_{10}(m_{\\text{rand}})$', '$\log_{10}(w)$', '$\log_{10}(a)$']
print(obs_pars)


def prior_fun(batch_size: int) -> np.ndarray:
    samples = []
    for _ in range(batch_size):
        samples.append(list(prior.rvs().values()))
    return np.array(samples)


def generate_population_data(param_batch: np.ndarray, cells_in_population: int, max_length: int) -> np.ndarray:
    """
    Generate population data
    :param param_batch:  batch of parameters
    :param cells_in_population:  number of cells in a population (50)
    :param max_length:  maximum length of the sequence
    :return:
    """
    data_batch = []
    for params in param_batch:
        params_dict = {key: p for key, p in zip(obs_pars.keys(), params)}
        sim = model.sample(params_dict)
        data_batch.append(sim)  # generates a cell population in one experiment

    data_batch_transformed = np.ones((param_batch.shape[0], cells_in_population, max_length, 3)) * np.nan
    # each cell is of different length, each with x and y coordinates, make a tensor out of it
    n_cells_not_visible = 0
    for p_id, population_sim in enumerate(data_batch):
        if len(population_sim) == 0:
            # no cells were visible in the simulation
            n_cells_not_visible += 1
            continue
        for c_id, cell_sim in enumerate(population_sim):
            # pre-pad the data with zeros, but first write zeros as nans to compute the mean and std
            data_batch_transformed[p_id, c_id, -len(cell_sim['x']):, 0] = cell_sim['x']
            data_batch_transformed[p_id, c_id, -len(cell_sim['y']):, 1] = cell_sim['y']
            data_batch_transformed[p_id, c_id, -len(cell_sim['y']):, 2] = cell_sim['t']

    if n_cells_not_visible > 0:
        print(f'Simulation with no cells visible: {n_cells_not_visible}/{len(data_batch)}')
    return data_batch_transformed


# %%

presimulation_path = '/home/jarruda_hpc/CellMigration/presimulations'
n_val_data = 100
cells_in_population = 143
n_params = len(obs_pars)
batch_size = 32
iterations_per_epoch = 100
# 1000 batches to be generated, 10 epochs until the batch is used again -> 32.000 simulations
epochs = 500

# check if gpu is available
print('gpu:', tf.config.list_physical_devices('GPU'))

bayesflow_prior = Prior(batch_prior_fun=prior_fun, param_names=param_names)
bayes_simulator = Simulator(batch_simulator_fun=partial(generate_population_data,
                                                        cells_in_population=cells_in_population,
                                                        max_length=max_sequence_length))
generative_model = GenerativeModel(prior=bayesflow_prior, simulator=bayes_simulator,
                                   skip_test=True,  # once is enough, simulation takes time
                                   name="Normalizing Flow Generative Model")
# %%
if presimulate:
    print('presimulating')
    from time import sleep
    sleep(job_array_id)

    # we create on batch per job and save it in a folder
    epoch_id = job_array_id // iterations_per_epoch
    generative_model.presimulate_and_save(
        batch_size=batch_size,
        folder_path=presimulation_path+f'/epoch_{epoch_id}',
        iterations_per_epoch=1,
        epochs=1,
        extend_from=job_array_id,
        disable_user_input=True
    )
    print('Done!')
    exit()

# %%

if os.path.exists(os.path.join(gp, 'validation_data.pickle')):
    with open(os.path.join(gp, 'validation_data.pickle'), 'rb') as f:
        valid_data = pickle.load(f)
else:
    print('Generating validation data')
    valid_data = generative_model(n_val_data)
    # save the data
    with open(os.path.join(gp, 'validation_data.pickle'), 'wb') as f:
        pickle.dump(valid_data, f)

x_mean = np.nanmean(valid_data['sim_data'], axis=(0, 1, 2))
x_std = np.nanstd(valid_data['sim_data'], axis=(0, 1, 2))
p_mean = np.mean(valid_data['prior_draws'], axis=0)
p_std = np.std(valid_data['prior_draws'], axis=0)
print('Mean and std of parameters:', p_mean, p_std)

if not training:
    exit()


trainer = load_model(
    model_id=job_array_id,
    x_mean=x_mean,
    x_std=x_std,
    p_mean=p_mean,
    p_std=p_std,
    generative_model=generative_model
)

if not os.path.exists(trainer.checkpoint_path):
    trainer._setup_optimizer(
        optimizer=None,
        epochs=epochs,
        iterations_per_epoch=iterations_per_epoch
    )

    history = trainer.train_from_presimulation(
        presimulation_path=presimulation_path,
        optimizer=trainer.optimizer,
        max_epochs=epochs,
        early_stopping=True,
        early_stopping_args={'patience': 17 - 2},
        custom_loader=custom_loader,
        validation_sims=valid_data
    )
    print('Training done!')
else:
    history = trainer.loss_history.get_plottable()

# diagnostic plots
valid_data_config = trainer.configurator(valid_data)

posterior_samples = trainer.amortizer.sample(valid_data_config, n_samples=100)
posterior_samples = posterior_samples * p_std + p_mean
prior_draws = valid_data_config["parameters"] * p_std + p_mean

_ = bf.diagnostics.plot_sbc_ecdf(posterior_samples, prior_draws, difference=True, param_names=log_param_names)
plt.savefig(f'{trainer.checkpoint_path}/sbc_ecdf.png')

posterior_samples = trainer.amortizer.sample(valid_data_config, n_samples=1000)
posterior_samples = posterior_samples * p_std + p_mean

_ = bf.diagnostics.plot_recovery(posterior_samples, prior_draws, param_names=log_param_names)
plt.savefig(f'{trainer.checkpoint_path}/recovery.png')


# simulate test data
for test_id in [0, 1, 2]:
    print('test', test_id)
    np.random.seed(test_id)
    test_params = np.array(list(prior.rvs().values()))
    if not os.path.exists(os.path.join(gp, f'test_sim_{test_id}.npy')):
        test_sim_full = bayes_simulator(test_params[np.newaxis])
        test_sim = test_sim_full['sim_data']
        np.save(os.path.join(gp, f'test_sim_{test_id}.npy'), test_sim)
    else:
        test_sim = np.load(os.path.join(gp, f'test_sim_{test_id}.npy'))
        test_sim_full = {'sim_data': test_sim}

    test_posterior_samples = trainer.amortizer.sample(trainer.configurator(test_sim_full), n_samples=100)
    test_posterior_samples = test_posterior_samples * p_std + p_mean

    # %%
    # get posterior samples and simulate
    posterior_sim = bayes_simulator(test_posterior_samples)['sim_data']

    # plot the summary statistics
    wasserstein_distance = plot_compare_summary_stats(test_sim, posterior_sim, path=f'{trainer.checkpoint_path}/{test_id}-Summary Stats')

    # plot the trajectories
    plot_trajectory(test_sim[0], posterior_sim[0], path=f'{trainer.checkpoint_path}/{test_id}-Simulations', show_umap=True)
    plot_autocorrelation(test_sim[0], posterior_sim[0], path=f'{trainer.checkpoint_path}/{test_id}-Autocorrelation')

    if trainer.amortizer.summary_loss is not None:
        test_data_config = trainer.configurator(test_sim_full)

        MMD_sampling_distribution, MMD_observed = trainer.mmd_hypothesis_test(
            observed_data=test_data_config,
            reference_data=valid_data_config,  # if not provided, will use the generative model
            num_null_samples=500,
            bootstrap=True  # if True, use the reference data as null samples
        )
        fig = bf.diagnostics.plot_mmd_hypothesis_test(MMD_sampling_distribution, MMD_observed)
        fig.savefig(f'{trainer.checkpoint_path}/{test_id}-Synthetic MMD.png', bbox_inches='tight')
        plt.show()

    print(f"Validation loss: {np.min(history['val_losses'])}")
    print(f"Wasserstein distance: {wasserstein_distance}")
del trainer
