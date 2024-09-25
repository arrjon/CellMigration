#%%
import math
import os
import pickle
from functools import partial
from datetime import timedelta
from typing import Union

import numpy as np
import pyabc
import scipy.stats as stats
from fitmulticell import model as morpheus_model
from fitmulticell.sumstat import SummaryStatistics

from summary_stats import reduced_coordinates_to_sumstat, reduce_to_coordinates, compute_mean_summary_stats

# get the job array id and number of processors
job_array_id = int(os.environ.get('SLURM_ARRAY_TASK_ID', 0))
n_procs = int(os.environ.get('SLURM_CPUS_PER_TASK', 1))
print(job_array_id)
on_cluster = True
population_size = 500
run_old_sumstats = True

if job_array_id == 1:
    run_old_sumstats = False

#%%
if on_cluster:
    gp = '/home/jarruda_hpc/CellMigration/synth_data_params_bayesflow'
else:
    gp = os.getcwd()

par_map = {
    'gradient_strength': './CellTypes/CellType/Constant[@symbol="gradient_strength"]',
    'move.strength': './CellTypes/CellType/Constant[@symbol="move.strength"]',
    'move.duration.scale': './CellTypes/CellType/Constant[@symbol="move.duration.scale"]',
    'cell_nodes': './Global/Constant[@symbol="cell_nodes"]',
}

model_path = gp + "/cell_movement_v23.xml"  # time step is 30sec, move.dir completely normalized, init move.dir rand in all directions
# defining the summary statistics function
max_sequence_length = 120
min_sequence_length = 0
only_longest_traj_per_cell = True  # mainly to keep the data batchable
cells_in_population = 50


def make_sumstat_dict(data: Union[dict, np.ndarray]) -> dict:
    if isinstance(data, dict):
        # get key
        key = list(data.keys())[0]
        data = data[key]
    data = data[0]  # only one sample
    # compute the summary statistics
    summary_stats_dict = reduced_coordinates_to_sumstat(data)
    (ad_mean, msd_mean, ta_mean, vel_mean, wt_mean,
            _, _, _, _, _) = compute_mean_summary_stats([summary_stats_dict], remove_nan=False)
    cleaned_dict = {
        'ad': np.array(ad_mean).flatten(),
        'msd': np.array(msd_mean).flatten(),
        'ta': np.array(ta_mean).flatten(),
        'vel': np.array(vel_mean).flatten(),
        'wt': np.array(wt_mean).flatten()
    }
    return cleaned_dict


def prepare_sumstats(output_morpheus_model) -> dict:
    sim_coordinates = reduce_to_coordinates(output_morpheus_model, 
                          minimal_length=min_sequence_length, 
                          maximal_length=max_sequence_length,
                          only_longest_traj_per_cell=only_longest_traj_per_cell
                          )
    
    # we now do exactly the same as in the BayesFlow workflow, but here we get only one sample at a time
    data_transformed = np.ones((1, cells_in_population, max_sequence_length, 2)) * np.nan
    # each cell is of different length, each with x and y coordinates, make a tensor out of it
    n_cells_not_visible = 0
    if len(sim_coordinates) != 0:
        # some cells were visible in the simulation
        for c_id, cell_sim in enumerate(sim_coordinates):
            # pre-pad the data with zeros, but first write zeros as nans to compute the mean and std
            data_transformed[0, c_id, -len(cell_sim['x']):, 0] = cell_sim['x']
            data_transformed[0, c_id, -len(cell_sim['y']):, 1] = cell_sim['y']
    
    return {'sim': data_transformed}


sumstat = SummaryStatistics(sum_stat_calculator=prepare_sumstats)                    

if on_cluster:
    # define the model object
    model = morpheus_model.MorpheusModel(
        model_path, par_map=par_map, par_scale="lin",
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
    'gradient_strength': 500000.,  # strength of the gradient of chemotaxis
    'move.strength': 1.,  # strength of directed motion
    'move.duration.scale': 30.,  # median of exponential distribution
    'cell_nodes': 30.,  # volume of the cell  # todo: unit? \mu mu^3?
}

# define parameters' limits
obs_pars_log = {key: math.log10(val) for key, val in obs_pars.items()}
limits = {'gradient_strength': (math.log10(10 ** 4), math.log10(10 ** 12)), 
          'move.strength': (0, math.log10(10 ** 5)),
          'move.duration.scale': (math.log10((10 ** -2) * 30), math.log10((10 ** 2) * 30)),
          'cell_nodes': (math.log10(10 ** 0), math.log10(10 ** 2))}


prior = pyabc.Distribution(**{key: pyabc.RV("uniform", lb, ub - lb)
                              for key, (lb, ub) in limits.items()})
param_names = list(obs_pars.keys())
print(obs_pars)
#%%
# simulate test data
test_params = np.log10(list(obs_pars.values()))
if not os.path.exists('test_sim.npy'):
    raise FileNotFoundError('Test data not found')
else:
    test_sim = np.load('test_sim.npy')
#%%
def obj_func_wass(sim: dict, obs: dict):
    total = 0
    for key in sim:
        x, y = np.array(sim[key]), np.array(obs[key])
        if x.size == 0:
            return np.inf
        total += stats.wasserstein_distance(x, y)
    return total
#%%
if run_old_sumstats:
    abc = pyabc.ABCSMC(model, prior,
                       distance_function=obj_func_wass,
                       summary_statistics=make_sumstat_dict,
                       population_size=population_size,
                       sampler=pyabc.sampler.MulticoreEvalParallelSampler(n_procs=n_procs))

    #db_path = os.path.join(tempfile.gettempdir(), "test.db")
    db_path = os.path.join(gp, "synthetic_test_old_sumstats.db")
    history = abc.new("sqlite:///" + db_path, make_sumstat_dict(test_sim))

    #start the abc fitting
    abc.run(min_acceptance_rate=1e-2, max_nr_populations=30, max_walltime=timedelta(days=6))
    print('Done!')
    exit()

#%%
import tensorflow as tf
from bayesflow.amortizers import AmortizedPosterior
from bayesflow.networks import InvertibleNetwork
from bayesflow.helper_networks import MultiConv1D
from bayesflow.trainers import Trainer
from bayesflow import default_settings as defaults
from tensorflow.keras.layers import Dense, GRU, LSTM, Bidirectional
from tensorflow.keras.models import Sequential
#%%
if os.path.exists(os.path.join(gp, 'validation_data.pickle')):
    with open(os.path.join(gp, 'validation_data.pickle'), 'rb') as f:
        valid_data = pickle.load(f)
else:
    raise FileNotFoundError('Validation data not found')

x_mean = np.nanmean(valid_data['sim_data'], axis=(0, 1, 2))
x_std = np.nanstd(valid_data['sim_data'], axis=(0, 1, 2))
p_mean = np.mean(valid_data['prior_draws'], axis=0)
p_std = np.std(valid_data['prior_draws'], axis=0)

# compute the mean of the summary statistics
summary_stats_list_ = [reduced_coordinates_to_sumstat(t) for t in valid_data['sim_data']]
(_, _, _, _, _, ad_averg, MSD_averg, 
 TA_averg, VEL_averg, WT_averg) = compute_mean_summary_stats(summary_stats_list_, remove_nan=False)
direct_conditions_ = np.stack([ad_averg, MSD_averg, TA_averg, VEL_averg, WT_averg]).T
# replace inf with -1
direct_conditions_[np.isinf(direct_conditions_)] = np.nan
        
summary_valid_max = np.nanmax(direct_conditions_, axis=0)
summary_valid_min = np.nanmin(direct_conditions_, axis=0)
#%%
def configurator(forward_dict: dict, remove_nans: bool = False, manual_summary: bool = False) -> dict:
    out_dict = {}

    # Extract data
    x = forward_dict["sim_data"]
    
    if remove_nans:
        # check if simulation with only nan values in a row
        non_nan_populations = np.isnan(x).sum(axis=(1,2,3))-np.prod(x.shape[1:]) != 0
        #print(x.shape[0]-non_nan_populations.sum(), 'samples with only nan values in a row')
        x = x[non_nan_populations]
    
    # compute manual summary statistics
    if manual_summary:
        summary_stats_list = [reduced_coordinates_to_sumstat(t) for t in x]
        # compute the mean of the summary statistics
        (_, _, _, _, _, 
         ad_averg, MSD_averg, 
         TA_averg, VEL_averg, WT_averg) = compute_mean_summary_stats(summary_stats_list, remove_nan=False)
        direct_conditions = np.stack([ad_averg, MSD_averg, TA_averg, VEL_averg, WT_averg]).T   
        # normalize statistics
        direct_conditions = (direct_conditions - summary_valid_min) / (summary_valid_max - summary_valid_min)
        # replace nan or inf with -1
        direct_conditions[np.isinf(direct_conditions)] = -1
        direct_conditions[np.isnan(direct_conditions)] = -1
        out_dict['direct_conditions'] = direct_conditions.astype(np.float32)
    
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


#%%
import logging
#%%
def load_model():
    # Set the logger to the desired level
    tf.get_logger().setLevel('ERROR')  # This will suppress warnings and info logs from TensorFlow

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
            self.conv_settings = conv_settings
    
            self.num_conv_layers = num_conv_layers
            self.conv = Sequential([MultiConv1D(conv_settings) for _ in range(num_conv_layers)])
            self.use_attention = use_attention
            
            self.use_lstm = use_lstm
            self.rnn_units = rnn_units
            self.bidirectional = bidirectional
            if use_lstm:
                self.rnn = Bidirectional(LSTM(rnn_units, return_sequences=use_attention)) if bidirectional else LSTM(rnn_units, return_sequences=use_attention)
            else:
                self.rnn = GRU(LSTM(rnn_units, return_sequences=use_attention)) if bidirectional else LSTM(rnn_units, return_sequences=use_attention)
    
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
        
        def get_config(self):
            """Return the config for serialization."""
            config = super().get_config()
            config.update({
                'summary_dim': self.summary_dim,
                'num_conv_layers': self.num_conv_layers,
                'rnn_units': self.rnn_units,
                'use_lstm': self.use_lstm,
                'bidirectional': self.bidirectional,
                'conv_settings': self.conv_settings,
                'use_attention': self.use_attention
            })
            return config
        
        @classmethod
        def from_config(cls, config):
            """Recreate the model from the config."""
            return cls(**config)
    
    job_array_id = 6
    
    num_coupling_layers = 6
    num_dense = 3
    use_attention = False
    use_bidirectional = False
    summary_loss = None
    use_manual_summary = False
    if job_array_id == 0:
        checkpoint_path = 'amortizer-cell-migration-conv-6'
        map_idx_sim = 21
    elif job_array_id == 1:
        checkpoint_path = 'amortizer-cell-migration-attention-6-bid'
        use_attention = True
        use_bidirectional = True
        map_idx_sim = 77
    elif job_array_id == 2:
        checkpoint_path = 'amortizer-cell-migration-conv-7'
        num_coupling_layers = 7
        map_idx_sim = 51
    elif job_array_id == 3:
        checkpoint_path = 'amortizer-cell-migration-attention-7'
        num_coupling_layers = 7
        use_attention = True
        map_idx_sim = 64
    elif job_array_id == 4:
        checkpoint_path = 'amortizer-cell-migration-attention-7-bid'
        num_coupling_layers = 7
        use_attention = True
        use_bidirectional = True
        map_idx_sim = 38
    elif job_array_id == 5:
        checkpoint_path = 'amortizer-cell-migration-attention-7-bid-MMD'
        num_coupling_layers = 7
        use_attention = True
        use_bidirectional = True
        summary_loss = 'MMD'
        map_idx_sim = 77
    elif job_array_id == 6:
        checkpoint_path = 'amortizer-cell-migration-attention-7-bid-MMD-manual'
        num_coupling_layers = 7
        use_attention = True
        use_bidirectional = True
        summary_loss = 'MMD'
        map_idx_sim = 57
        use_manual_summary = True
    else:
        raise ValueError('Checkpoint path not found')
    os.makedirs(f"../results/{checkpoint_path}", exist_ok=True)
    
    summary_net = GroupSummaryNetwork(summary_dim=len(obs_pars) * 2,
                                      rnn_units=2 ** int(np.ceil(np.log2(max_sequence_length))),
                                      use_attention=use_attention,
                                      bidirectional=use_bidirectional)
    inference_net = InvertibleNetwork(num_params=len(obs_pars),
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
    
    amortizer = AmortizedPosterior(inference_net=inference_net, summary_net=summary_net,
                                   summary_loss_fun=summary_loss)
    
    # Disable logging
    logging.disable(logging.CRITICAL)

    # build the trainer with networks
    max_to_keep = 17
    trainer = Trainer(amortizer=amortizer,
                      configurator=partial(configurator, 
                                           manual_summary=use_manual_summary),
                      checkpoint_path=checkpoint_path,
                      skip_checks=True,
                      max_to_keep=max_to_keep)
        
    # check if file exist
    if os.path.exists(checkpoint_path):
        trainer.load_pretrained_network()
        history = trainer.loss_history.get_plottable()
    else:
        raise FileNotFoundError('Checkpoint path not found')
    
    # Find the checkpoint with the lowest validation loss out of the last max_to_keep
    recent_losses = history['val_losses'].iloc[-max_to_keep:]
    best_valid_epoch = recent_losses['Loss'].idxmin() + 1  # checkpoints are 1-based indexed
    new_checkpoint = trainer.manager.latest_checkpoint.rsplit('-', 1)[0] + f'-{best_valid_epoch}'    
    trainer.checkpoint.restore(new_checkpoint)
    #print("Networks loaded from {}".format(new_checkpoint))
    
    # Re-enable logging
    logging.disable(logging.NOTSET)
    
    return summary_net, use_manual_summary
#%%
_ = load_model()


#%%
# use trained neural net as summary statistics
def make_sumstat_dict_nn(
        data: Union[dict, np.ndarray], 
        #summary_nn: keras.models, 
        config_map: callable,
        #manual_summary: bool
) -> dict:
    if isinstance(data, dict):
        # get key
        key = list(data.keys())[0]
        data = data[key]
        
    summary_nn, manual_summary = load_model()
    
    # configures the input for the network
    config_input = config_map({"sim_data": data}, manual_summary=manual_summary)
    # get the summary statistics
    out_dict = {
        'summary_net': summary_nn(config_input['summary_conditions']).numpy().flatten()
    }
    # if direct conditions are available, concatenate them
    if 'direct_conditions' in config_input.keys():
        out_dict['direct_conditions'] = config_input['direct_conditions'].flatten() 
    return out_dict

sumstats_nn = partial(make_sumstat_dict_nn,
                      config_map=configurator)

if on_cluster:
    # define the model object
    model_nn = morpheus_model.MorpheusModel(
        model_path, par_map=par_map, par_scale="log10",
        show_stdout=False, show_stderr=False,
        executable="ulimit -s unlimited; /home/jarruda_hpc/CellMigration/morpheus-2.3.7",
        clean_simulation=True,
        raise_on_error=False, sumstat=sumstat)

    # todo: remember also change tiff path in model.xml!
else:
    # define the model object
    model_nn = morpheus_model.MorpheusModel(
        model_path, par_map=par_map, par_scale="log10",
        show_stdout=False, show_stderr=False,
        clean_simulation=True,
        raise_on_error=False, sumstat=sumstat)
#%%
synthetic_data_test_nn = sumstats_nn(test_sim)

#%%
abc = pyabc.ABCSMC(model_nn, prior, # here we use now the Euclidean distance
                   population_size=population_size,
                   summary_statistics=sumstats_nn,
                   sampler=pyabc.sampler.MulticoreEvalParallelSampler(n_procs=n_procs))
#db_path = os.path.join(tempfile.gettempdir(), "test.db")
db_path = os.path.join(gp, "synthetic_test_new_sumstats.db")
history = abc.new("sqlite:///" + db_path, sumstats_nn(test_sim))

#start the abc fitting
abc.run(min_acceptance_rate=1e-2, max_nr_populations=30, max_walltime=timedelta(days=6))
#%%
print('Done!')
