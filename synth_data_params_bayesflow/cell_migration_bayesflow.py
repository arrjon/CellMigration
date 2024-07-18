# %%
import math
import os
import pickle
import sys
from functools import partial

import numpy as np
import pyabc
from fitmulticell import model as morpheus_model
from fitmulticell.sumstat import SummaryStatistics

from summary_stats import reduce_to_coordinates

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
    #'move.duration.median': './CellTypes/CellType/Constant[@symbol="move.duration.median"]',
    #    'move.duration.scale': './CellTypes/CellType/Constant[@symbol="move.duration.scale"]',
    'cell_nodes': './Global/Constant[@symbol="cell_nodes"]',
    #'d_env': './CellTypes/CellType/Property[@symbol="d_env"]',
}

model_path = gp + "/cell_movement_v23.xml"  # time step is 30sec
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
        model_path, par_map=par_map, par_scale="lin",
        show_stdout=False, show_stderr=False,
        executable="ulimit -s unlimited; /home/jarruda_hpc/CellMigration/morpheus-2.3.7",
        raise_on_error=False, sumstat=sumstat)

    # todo: remember also change tiff path in model.xml!
else:
    # define the model object
    model = morpheus_model.MorpheusModel(
        model_path, par_map=par_map, par_scale="lin",
        show_stdout=False, show_stderr=False,
        raise_on_error=False, sumstat=sumstat)

# parameter values used to generate the synthetic data
obs_pars = {
    'gradient_strength': 500000,  # strength of the gradient of chemotaxis
    'move.strength': 0.021,  # strength of directed motion
    'move.duration.mean': 100,  # mean waiting time of exponential distribution
    #'move.duration.median': 972.33,  # median waiting time of gamma distribution (actually the mean)
    # rand_gamma(move.duration.median/move.duration.scale,move.duration.scale)
    # 'move.duration.scale': 96899.44860996476,
    'cell_nodes': 30,  # volume of the cell
    #'d_env': 0.01,  # influence of random motion vs biased direction (previous movement?)
    # d_env * move.dir.x / move.dir.abs + (1-d_env)*cos(alpha), d_env * move.dir.y / move.dir.abs + (1-d_env)*sin(alpha) , 0
}

# define the parameter scale
model.par_scale = "log10"
# define parameters' limits
obs_pars_log = {key: math.log10(val) for key, val in obs_pars.items()}
limits = {key: (math.log10((10 ** -3) * val), math.log10((10 ** 3) * val)) for
          key, val in obs_pars.items()}

#limits['d_env'] = (math.log10(10 ** -5), math.log10(10 ** 0))
limits['cell_nodes'] = (math.log10(10 ** 0), math.log10(10 ** 2))

prior = pyabc.Distribution(**{key: pyabc.RV("uniform", lb, ub - lb)
                              for key, (lb, ub) in limits.items()})
param_names = list(obs_pars.keys())
# %%
import tensorflow as tf
from bayesflow.amortizers import AmortizedPosterior
from bayesflow.networks import InvertibleNetwork
from bayesflow.helper_networks import MultiConv1D
from bayesflow.simulation import GenerativeModel, Prior, Simulator
from bayesflow.trainers import Trainer
from bayesflow import default_settings as defaults
from functools import partial
from tensorflow.keras.layers import Dense, GRU, LSTM, Bidirectional
from tensorflow.keras.models import Sequential


def prior_fun(batch_size: int) -> np.ndarray:
    samples = []
    for _ in range(batch_size):
        samples.append(list(prior.rvs().values()))
    return np.array(samples)


def generate_population_data(param_batch: np.ndarray, data_dim: int, max_length: int) -> np.ndarray:
    """
    Generate population data
    :param param_batch:  batch of parameters
    :param data_dim:  number of cells in a population (50)
    :param max_length:  maximum length of the sequence
    :return:
    """
    data_raw = []
    for params in param_batch:
        params_dict = {key: p for key, p in zip(obs_pars.keys(), params)}
        sim = model.sample(params_dict)
        data_raw.append(sim)  # generates a cell population in one experiment

    data = np.ones((param_batch.shape[0], data_dim, max_length, 2)) * np.nan
    # each cell is of different length, each with x and y coordinates, make a tensor out of it
    n_cells_not_visible = 0
    for s_id, sample in enumerate(data_raw):
        if len(sample) == 0:
            # no cells were visible in the simulation
            n_cells_not_visible += 1
            continue
        for c_id, cell in enumerate(sample):
            # pre-pad the data with zeros, but first write zeros as nans to compute the mean and std
            data[s_id, c_id, -len(cell['x']):, 0] = cell['x']
            data[s_id, c_id, -len(cell['y']):, 1] = cell['y']

    print(f'Samples with no cells visible: {n_cells_not_visible}/{len(data_raw)}')
    return data


# %%

presimulation_path = 'presimulations'
n_val_data = 100
data_dim = 50
n_params = len(obs_pars)
batch_size = 32
iterations_per_epoch = 100
# 4000 batches to be generated, 40 epochs until the batch is used again
epochs = 500

# check if gpu is available
print('gpu:', tf.config.list_physical_devices('GPU'))

bayesflow_prior = Prior(batch_prior_fun=prior_fun, param_names=param_names)
bayes_simulator = Simulator(batch_simulator_fun=partial(generate_population_data, data_dim=data_dim,
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
    generative_model.presimulate_and_save(batch_size=batch_size,
                                          folder_path=presimulation_path + f'/epoch_{epoch_id}',
                                          iterations_per_epoch=1,
                                          epochs=1,
                                          extend_from=job_array_id,
                                          disable_user_input=True)
    print('Done!')
    exit()


# %%

def custom_loader(file_path):
    """Uses pickle to load, but each path is folder with multiple files, each one batch"""
    # load all files in folder
    loaded_presimulations = []
    for file in os.listdir(file_path):
        with open(os.path.join(file_path, file), 'rb') as f:
            test = pickle.load(f)[0]
            assert isinstance(test, dict)
            loaded_presimulations.append(test)
    # shuffle list, so iterations are random, only batches stay the same
    np.random.shuffle(loaded_presimulations)
    return loaded_presimulations


# %%

if os.path.exists(os.path.join(gp, 'validation_data.pickle')):
    with open(os.path.join(gp, 'validation_data.pickle'), 'rb') as f:
        sim_data = pickle.load(f)
else:
    print('Generating validation data')
    sim_data = generative_model(n_val_data)
    # save the data
    with open(os.path.join(gp, 'validation_data.pickle'), 'wb') as f:
        pickle.dump(sim_data, f)

x_mean = np.nanmean(sim_data['sim_data'], axis=(0, 1, 2))
x_std = np.nanstd(sim_data['sim_data'], axis=(0, 1, 2))
p_mean = np.mean(sim_data['prior_draws'], axis=0)
p_std = np.std(sim_data['prior_draws'], axis=0)
print('Mean and std of data:', x_mean, x_std)
print('Mean and std of parameters:', p_mean, p_std)

if not training:
    exit()


def configurator(forward_dict: dict, remove_nans: bool) -> dict:
    out_dict = {}

    # Extract data
    x = forward_dict["sim_data"]

    if remove_nans:
        # check if simulation with only nan values in a row
        non_nan_populations = np.isnan(x).sum(axis=(1, 2, 3)) - np.prod(x.shape[1:]) != 0
        # print(x.shape[0]-non_nan_populations.sum(), 'samples with only nan values in a row')
        x = x[non_nan_populations]

    # Normalize data
    x = (x - x_mean) / x_std

    # Check for NaN values in the first entry of the last axis
    # If nan_mask is False (no NaNs), set to 1; otherwise, set to 0
    nan_mask = np.isnan(x[..., 0])
    new_dim = np.where(nan_mask, 0, 1)
    new_dim_expanded = np.expand_dims(new_dim, axis=-1)
    x = np.concatenate((x, new_dim_expanded), axis=-1)

    # Normalize data
    x[np.isnan(x)] = 0  # replace nan with 0, pre-padding (since we have nans in the data at the end)
    out_dict['summary_conditions'] = x.astype(np.float32)

    # Extract params
    if 'parameters' in forward_dict.keys():
        forward_dict["prior_draws"] = forward_dict["parameters"]
    if 'prior_draws' in forward_dict.keys():
        params = forward_dict["prior_draws"]
        if remove_nans:
            params = params[non_nan_populations]
        params = (params - p_mean) / p_std
        out_dict['parameters'] = params.astype(np.float32)
    return out_dict


# define the network
class GroupSummaryNetwork(tf.keras.Model):
    """Network to summarize the data of groups of cells.  Each group is passed through a series of convolutional layers
    followed by an LSTM layer. The output of the LSTM layer is then pooled across the groups and dense layer applied
    to obtain a summary of fixed dimensionality. The network is invariant to the order of the groups.
    """

    def __init__(
            self,
            summary_dim,
            num_conv_layers=2,
            rnn_units=128,
            use_lstm=True,
            bidirectional=False,
            conv_settings=None,
            use_attention=False,
            **kwargs
    ):
        super().__init__(**kwargs)

        if conv_settings is None:
            conv_settings = defaults.DEFAULT_SETTING_MULTI_CONV

        self.conv = Sequential([MultiConv1D(conv_settings) for _ in range(num_conv_layers)])
        self.use_attention = use_attention

        if use_lstm:
            self.rnn = Bidirectional(LSTM(rnn_units, return_sequences=use_attention)) if bidirectional else LSTM(
                rnn_units, return_sequences=use_attention)
        else:
            self.rnn = GRU(LSTM(rnn_units, return_sequences=use_attention)) if bidirectional \
                else LSTM(rnn_units, return_sequences=use_attention)

        if self.use_attention:
            self.attention = tf.keras.layers.Attention()
        self.out_layer = Dense(summary_dim, activation="linear")
        self.summary_dim = summary_dim

    def call(self, x, **kwargs):
        """Performs a forward pass through the network by first passing `x` through the same rnn network for
        each household and then pooling the outputs across households.

        Parameters
        ----------
        x : tf.Tensor
            Input of shape (batch_size, n_groups, n_time_steps, n_features)

        Returns
        -------
        out : tf.Tensor
            Output of shape (batch_size, summary_dim)
        """
        # iterate over groups
        out_list = []  # list to store outputs of LSTM for each group
        for g_i in range(x.shape[1]):
            out = self.conv(x[:, g_i], **kwargs)  # (batch_size, n_time_steps, n_filters) -> default: filters=32
            out = self.rnn(out, **kwargs)  # (batch_size, lstm_units)
            # if attention is used, return full sequence (batch_size, n_time_steps, lstm_units)
            # bidirectional LSTM returns 2*lstm_units
            out_list.append(out)
        if self.use_attention:
            # learn a query vector to attend over the groups, some groups might be more important
            # this should be invariant to the order of the groups (depends on the learned attention mechanism)
            out = tf.stack(out_list, axis=1)  # (batch_size, n_groups, n_time_steps, lstm_units)
            query = tf.reduce_mean(out, axis=2)  # (batch_size, n_groups, lstm_units)
            # Reshape query to match the required shape for attention
            query = tf.expand_dims(query, axis=2)  # (batch_size, n_groups, 1, lstm_units)
            out = self.attention([query, out], **kwargs)  # (batch_size, n_groups, 1, lstm_units)
            out = tf.reduce_max(out, axis=1)  # (batch_size, 1, lstm_units)
            out = tf.squeeze(out, axis=1)  # Remove the extra dimension (batch_size, lstm_units)
        else:
            # max pooling over groups, this totally invariants to the order of the groups
            out = tf.reduce_max(out_list, axis=0)  # (batch_size, lstm_units)
        # apply dense layer
        out = self.out_layer(out, **kwargs)  # (batch_size, summary_dim)
        return out


checkpoint_path = 'amortizer-cell-migration-6'
num_coupling_layers = 6
num_dense = 2
use_attention = False
use_bidirectional = False
config_remove_nans = False
if job_array_id == 0:
    checkpoint_path = 'amortizer-cell-migration-conv-7'
    num_coupling_layers = 7
    num_dense = 3
elif job_array_id == 1:
    checkpoint_path = 'amortizer-cell-migration-attention-7'
    num_coupling_layers = 7
    num_dense = 3
    use_attention = True
elif job_array_id == 2:
    checkpoint_path = 'amortizer-cell-migration-attention-7-bid'
    num_coupling_layers = 7
    num_dense = 3
    use_attention = True
    use_bidirectional = True
# elif checkpoint_path == 'amortizer-cell-migration-conv-7-nan':
#     num_coupling_layers = 7
#     num_dense = 3
#     config_remove_nans = True
# elif checkpoint_path == 'amortizer-cell-migration-attention-7-nan':
#     num_coupling_layers = 7
#     num_dense = 3
#     use_attention = True
#     config_remove_nans = True
# elif checkpoint_path == 'amortizer-cell-migration-attention-7-bid-nan':
#     num_coupling_layers = 7
#     num_dense = 3
#     use_attention = True
#     use_bidirectional = True
#     config_remove_nans = True
else:
    raise ValueError('Checkpoint path not found')
print(checkpoint_path)

summary_net = GroupSummaryNetwork(summary_dim=n_params * 2,
                                  rnn_units=2 ** int(np.ceil(np.log2(max_sequence_length))),
                                  use_attention=use_attention,
                                  bidirectional=use_bidirectional)
inference_net = InvertibleNetwork(num_params=n_params,
                                  num_coupling_layers=num_coupling_layers,
                                  coupling_design='spline',
                                  coupling_settings={
                                      "num_dense": num_dense,
                                      "dense_args": dict(
                                          activation='relu',
                                          kernel_regularizer=tf.keras.regularizers.l2(1e-4),
                                      ),
                                      "dropout_prob": 0.2,
                                      "bins": 16,
                                  })

amortizer = AmortizedPosterior(inference_net=inference_net, summary_net=summary_net)

# %%
# build the trainer with networks and generative model
max_to_keep = 17
trainer = Trainer(amortizer=amortizer,
                  configurator=partial(configurator, remove_nans=config_remove_nans),
                  generative_model=generative_model,
                  checkpoint_path=checkpoint_path,
                  skip_checks=True,  # once is enough, simulation takes time
                  max_to_keep=max_to_keep)

# check if file exist
if os.path.exists(checkpoint_path):
    trainer.load_pretrained_network()
    history = trainer.loss_history.get_plottable()
else:
    trainer._setup_optimizer(optimizer=None,
                             epochs=epochs,
                             iterations_per_epoch=iterations_per_epoch)

    history = trainer.train_from_presimulation(presimulation_path=presimulation_path,
                                               optimizer=trainer.optimizer,
                                               max_epochs=epochs,
                                               early_stopping=True,
                                               early_stopping_args={'patience': max_to_keep-2},
                                               custom_loader=custom_loader,
                                               validation_sims=sim_data)
    print('Training done!')

# Check if training converged
if np.isnan(history['val_losses'].iloc[-1]).any():
    print('Training failed with NaN loss at the end')
    if np.isnan(history['val_losses'].iloc[-max_to_keep:]).all():
        print('Training failed with NaN loss for all latest checkpoints')

# Find the checkpoint with the lowest validation loss out of the last max_to_keep
recent_losses = history['val_losses'].iloc[-max_to_keep:]
best_valid_epoch = recent_losses['Loss'].idxmin() + 1  # checkpoints are 1-based indexed
new_checkpoint = trainer.manager.latest_checkpoint.rsplit('-', 1)[0] + f'-{best_valid_epoch}'
trainer.checkpoint.restore(new_checkpoint)
print("Networks loaded from {}".format(new_checkpoint))

# simulate test data
test_params = np.log10(list(obs_pars.values()))
if not os.path.exists('test_sim.npy'):
    test_sim_full = bayes_simulator(test_params[np.newaxis])
    test_sim = test_sim_full['sim_data']
    np.save('test_sim.npy', test_sim)
else:
    test_sim = np.load('test_sim.npy')
    test_sim_full = {'sim_data': test_sim}

test_posterior_samples = amortizer.sample(trainer.configurator(test_sim_full), n_samples=100)
test_posterior_samples = test_posterior_samples * p_std + p_mean

# compute the log posterior of the test data
input_dict = {
    'sim_data': np.repeat(test_sim, repeats=100, axis=0),
    'parameters': test_posterior_samples
}
log_prob = amortizer.log_posterior(trainer.configurator(input_dict))

# get the MAP
map_idx = np.argmax(log_prob)

# %%
# get posterior samples and simulate
if not os.path.exists(checkpoint_path + '/posterior_sim.npy'):
    # simulate the data
    posterior_sim = bayes_simulator(test_posterior_samples)['sim_data']
    np.save(checkpoint_path + '/posterior_sim.npy', posterior_sim)

    print('map_sim', map_idx, log_prob[map_idx], test_posterior_samples[map_idx])
