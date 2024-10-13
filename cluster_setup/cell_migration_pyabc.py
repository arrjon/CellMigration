#%%
import argparse
import logging
import math
import os
import pickle
from functools import partial
from typing import Optional
from typing import Union

import numpy as np
import pandas as pd
import pyabc
import scipy.stats as stats
import tensorflow as tf
import tidynamics  # to get sliding history stats in N*logN instead of N^2
from bayesflow import default_settings as defaults
from bayesflow.amortizers import AmortizedPosterior
from bayesflow.helper_networks import MultiConv1D
from bayesflow.networks import InvertibleNetwork
from bayesflow.trainers import Trainer
from fitmulticell import model as morpheus_model
from fitmulticell.sumstat import SummaryStatistics
from pyabc.sampler import RedisEvalParallelSampler
from tensorflow.keras.layers import Dense, GRU, LSTM, Bidirectional
from tensorflow.keras.models import Sequential


# defining the summary statistics functions
def turning_angle(data_dict):
    """Compute the angle between two consecutive points."""
    x = data_dict['x']
    y = data_dict['y']
    vx = np.diff(x)
    vy = np.diff(y)
    theta = np.arctan2(vy, vx)
    theta = np.unwrap(theta)
    theta = np.diff(theta)
    return theta


def velocity(data_dict):
    """Compute the velocity of the cell."""
    x = data_dict['x']
    y = data_dict['y']
    vx = np.diff(x)
    vy = np.diff(y)
    v = np.sqrt(vx ** 2 + vy ** 2)
    return v


def MSD(data_dict, x_name="x", y_name="y"):
    """Compute the mean square displacement of the cell."""
    msd = tidynamics.msd(
        np.column_stack([data_dict[x_name], data_dict[y_name]]))
    return msd


def angle_degree(data_dict):
    """Compute the absolute angle between two consecutive points in degrees with respect to the x-axis."""
    x = data_dict['x']
    y = data_dict['y']
    vx = np.diff(x)
    vy = np.diff(y)
    list_angle_degrees = []
    for x, y in zip(vx, vy):
        list_angle_degrees.append(math.degrees(math.atan2(x, y)))
    return list_angle_degrees


def mean_waiting_time(data_dict, time_interval=30., threshold=np.pi/4):
    """Compute the mean waiting time of the cell until it changes direction."""
    cell = np.stack([data_dict['x'], data_dict['y']], axis=1)
    time_steps = len(data_dict['x'])
    waiting_times = []
    last_change = 0

    # Compute initial direction
    initial_direction = cell[1] - cell[0]
    last_direction = np.arctan2(initial_direction[1], initial_direction[0])

    for t in range(2, time_steps):
        # Compute current direction
        current_vector = cell[t] - cell[t-1]
        current_direction = np.arctan2(current_vector[1], current_vector[0])

        # Check if direction has changed
        if abs(current_direction - last_direction) > threshold:
            waiting_times.append((t-1) - last_change)
            last_change = t-1

        last_direction = current_direction

    if waiting_times:
        mean_wt = np.mean(waiting_times) * time_interval
    else:
        mean_wt = np.nan  # If no direction change occurred
    return mean_wt


# my functions
def cut_region(data_dict, x_min, x_max, y_min, y_max, return_longest) -> Optional[list[dict]]:
    """
    Cut the region of interest from the data.
    Truncate the data to the longest list if 'return_longest' is True.
    """
    x = data_dict['x']
    y = data_dict['y']
    t = data_dict['t']
    obs_list = []
    x_cut = []
    y_cut = []
    t_cut = []
    entered = False
    for x_, y_, t_ in zip(x, y, t):
        if x_min < x_ < x_max and y_min < y_ < y_max:
            # if the cell is in the region
            entered = True
            x_cut.append(x_)
            y_cut.append(y_)
            t_cut.append(t_)
        elif entered:
            # if the cell has left the region
            entered = False
            # append dicts to the lists
            obs_list.append({'x': x_cut, 'y': y_cut, 't': t_cut})
            # empty the lists
            x_cut = []
            y_cut = []
            t_cut = []
        # if the cell is outside the region, do nothing

    # append the last cell if entered
    if entered and len(x_cut) > 0:
        obs_list.append({'x': x_cut, 'y': y_cut, 't': t_cut})

    if len(obs_list) == 0:
        return None

    if return_longest:
        # get the longest list
        max_id = np.argmax([len(d['x']) for d in obs_list])
        return [obs_list[max_id]]
    return obs_list


def compute_mean(x: Union[float, int, list, np.ndarray]) -> Union[float, int]:
    """
    Compute the mean if x is a non-empty list.
    Return x if it's a float/int.
    Return np.nan if it's an empty list or unrecognized type.
    """
    if isinstance(x, (list, np.ndarray)):
        return np.mean(x) if len(x) > 0 else np.nan
    elif isinstance(x, (int, float)):
        return x
    else:
        return np.nan


def compute_var(x: Union[float, int, list, np.ndarray]) -> Union[float, int]:
    """
    Compute the variance if x is a non-empty list.
    Return x if it's a float/int.
    Return np.nan if it's an empty list or unrecognized type.
    """
    if isinstance(x, (list, np.ndarray)):
        return np.var(x) if len(x) > 0 else np.nan
    elif isinstance(x, (int, float)):
        return 0.0
    else:
        return np.nan


def compute_autocorrelation(list_statistic: list) -> list:
    """
    Compute the autocorrelation of a list of statistics.
    """
    autocorr_results = []
    for s in list_statistic:
        # Convert to pandas Series to handle NaN values easily
        s_series = pd.Series(s)

        # Calculate the autocorrelation, skipping NaNs automatically
        autocorr = [s_series.autocorr(lag) for lag in range(len(list_statistic) + 1)]

        # Store the result for each cell
        autocorr_results.append(autocorr)
    return autocorr_results


def compute_summary_stats(cell_population: np.ndarray) -> (list, list, list, list, list):
    """
    Compute the statistics of the reduced/visible coordinates of each cell in a cell population.

    :param cell_population: 3D array of cell populations
    :return: list of msd, ta, v, ad, wt
    """
    msd_list = []
    ta_list = []
    v_list = []
    ad_list = []
    wt_list = []

    cell_count = cell_population.shape[0]
    # check if sim_dict_cut is empty
    for i in range(0, cell_count):
        # select another cell
        sim_dict = {'x': cell_population[i, :, 0], 'y': cell_population[i, :, 1]}
        if all(np.isnan(sim_dict['x'])):
            continue
        else:
            msd_list.append(MSD(sim_dict))
            ta_list.append(turning_angle(sim_dict))
            v_list.append(velocity(sim_dict))
            ad_list.append(angle_degree(sim_dict))
            wt_list.append(mean_waiting_time(sim_dict))

    return msd_list, ta_list, v_list, ad_list, wt_list


def reduced_coordinates_to_sumstat(cell_population: np.ndarray) -> dict:
    """
    Compute the summary statistics (mean per cell) of the reduced/visible coordinates of the cell population.

    :param cell_population: 3D array of cell populations
    :return: dictionary of summary statistics
    """
    msd_list, ta_list, v_list, ad_list, wt_list = compute_summary_stats(cell_population)

    if not msd_list:
        return {'msd_mean': [np.nan], 'msd_var': [np.nan],
                'ta_mean': [np.nan], 'ta_var': [np.nan],
                'v_mean': [np.nan], 'v_var': [np.nan],
                'ad_mean': [np.nan], 'ad_var': [np.nan],
                'wt_mean': [np.nan], 'wt_var': [np.nan]}

    # get the mean of list of lists with different lengths
    msd_mean = [compute_mean(x) for x in msd_list]
    ta_mean = [compute_mean(x) for x in ta_list]
    v_mean = [compute_mean(x) for x in v_list]
    ad_mean = [compute_mean(x)  for x in ad_list]
    wt_mean = [compute_mean(x) for x in wt_list]

    # get the variance of list of lists with different lengths
    msd_var = [compute_var(x) for x in msd_list]
    ta_var = [compute_var(x) for x in ta_list]
    v_var = [compute_var(x) for x in v_list]
    ad_var = [compute_var(x) for x in ad_list]
    wt_var = [compute_var(x) for x in wt_list]

    sim = {'msd_mean': msd_mean, 'msd_var': msd_var,
           'ta_mean': ta_mean, 'ta_var': ta_var,
           'v_mean': v_mean, 'v_var': v_var,
           'ad_mean': ad_mean, 'ad_var': ad_var,
           'wt_mean': wt_mean, 'wt_var': wt_var}
    return sim


def reduce_to_coordinates(sumstat: dict,
                          minimal_length: int = 0,
                          maximal_length: Optional[int] = None,
                          only_longest_traj_per_cell: bool = True) -> list[dict]:
    """
    Reduce the output of the model to the visible coordinates of the cells.

    :param sumstat: output of Morpheus model
    :param minimal_length: minimal length of the trajectory
    :param maximal_length: maximal length of the trajectory
    :param only_longest_traj_per_cell: if True, only the longest trajectory of each cell is returned
    :return: list of dictionaries with x, y, t coordinates
    """
    sim_list = []

    # get cell ids
    cell_ids = np.unique(sumstat['cell.id'])
    # check if sim_dict_cut is empty
    for cell_id in cell_ids:
        # select the next cell
        cell_data_idx = sumstat['cell.id'] == cell_id
        # retrieve the x and y coordinates
        # remove first element of x and y (there is waiting time, until we observe the cell)
        sim_dict = {
            'x': sumstat["cell.center.x"][cell_data_idx][1:],
            'y': sumstat["cell.center.y"][cell_data_idx][1:],
            't': sumstat["time"][cell_data_idx][1:]
        }
        # cut the region of interest, divide cells into multiple cells if they leave the region
        sim_dict_cut = cut_region(sim_dict,
                                  x_min=316.5, x_max=856.5, y_min=1145, y_max=1351,
                                  return_longest=only_longest_traj_per_cell)
        # check if sim_dict_cut is empty
        if sim_dict_cut is not None:
            sim_list.extend(sim_dict_cut)
        else:
            continue

    # only keep cells of minimal length
    if minimal_length > 0:
        sim_list = [sim for sim in sim_list if len(sim['x']) > minimal_length]
    if maximal_length is not None:
        sim_list = [{'x': sim['x'][:maximal_length],
                     'y': sim['y'][:maximal_length],
                     't': np.array(sim['t'][:maximal_length])
                     } for sim in sim_list]
    return sim_list


def clean_and_average(stat_list: list, remove_nan: bool):
    """
    Remove NaN and Inf from the list and compute the mean.
    """
    cleaned = [[x for x in stat if not np.isnan(x) and not np.isinf(x)] for stat in stat_list]
    averaged = [np.mean(stat) if len(stat) > 0 else np.nan for stat in cleaned]
    if remove_nan:
        averaged = [x for x in averaged if not np.isnan(x) and not np.isinf(x)]
    return cleaned, np.array(averaged)


def compute_mean_summary_stats(simulation_list: list[dict], remove_nan: bool = True) -> list:
    """
    Compute the mean summary statistics of the simulation list.

    param
        simulation_list: list of cell populations
        remove_nan: remove nan and inf values from the averaged summary statistics of a population
    :return:
    """

    # Extract the statistics from the simulations
    stat_keys = ['ad_mean', 'msd_mean', 'ta_mean', 'v_mean', 'wt_mean']
    result = []

    for key in stat_keys:
        stat_list = [sim[key] for sim in simulation_list]
        cleaned, averaged = clean_and_average(stat_list, remove_nan)
        result.extend([cleaned, averaged])

    return result


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


def configurator(forward_dict: dict,
                 x_mean: np.ndarray, x_std: np.ndarray,
                 p_mean: np.ndarray, p_std: np.ndarray,
                 summary_valid_min: np.ndarray = None, summary_valid_max: np.ndarray = None,
                 manual_summary: bool = False) -> dict:
    out_dict = {}

    # Extract data
    x = forward_dict["sim_data"]

    # compute manual summary statistics
    if manual_summary:
        summary_stats_list = [reduced_coordinates_to_sumstat(t) for t in x]
        # compute the mean of the summary statistics
        (_, ad_averg, _, MSD_averg, _,
         TA_averg, _, VEL_averg, _, WT_averg) = compute_mean_summary_stats(summary_stats_list, remove_nan=False)
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
            bidirectional=True,
            conv_settings=None,
            use_attention=False,
            return_attention_weights=False,
            use_GRU=True,
            **kwargs
    ):
        super().__init__(**kwargs)

        if conv_settings is None:
            conv_settings = defaults.DEFAULT_SETTING_MULTI_CONV
        self.conv_settings = conv_settings

        conv = Sequential([MultiConv1D(conv_settings) for _ in range(num_conv_layers)])
        self.num_conv_layers = num_conv_layers
        self.group_conv = tf.keras.layers.TimeDistributed(conv)
        self.use_attention = use_attention
        self.return_attention_weights = return_attention_weights
        self.pooling = tf.keras.layers.GlobalAveragePooling1D()
        self.rnn_units = rnn_units
        self.use_GRU = use_GRU
        self.bidirectional = bidirectional

        if self.use_attention:
            self.attention = tf.keras.layers.MultiHeadAttention(num_heads=4, key_dim=rnn_units)

        if use_GRU:
            rnn = Bidirectional(GRU(rnn_units, return_sequences=use_attention)) if bidirectional else GRU(rnn_units,
                                                                                                          return_sequences=use_attention)
        else:
            rnn = Bidirectional(LSTM(rnn_units, return_sequences=use_attention)) if bidirectional else LSTM(rnn_units,
                                                                                                            return_sequences=use_attention)
        self.group_rnn = tf.keras.layers.TimeDistributed(rnn)

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
        # Apply the RNN to each group
        out = self.group_conv(x, **kwargs)
        out = self.group_rnn(out, **kwargs)  # (batch_size, n_groups, lstm_units)
        # if attention is used, return full sequence (batch_size, n_groups, n_time_steps, lstm_units)
        # bidirectional LSTM returns 2*lstm_units

        if self.use_attention:
            # learn a query vector to attend over the time points
            query = tf.reduce_mean(out, axis=1)
            # Reshape query to match the required shape for attention
            query = tf.expand_dims(query, axis=1)  # (batch_size, 1, n_time_steps, lstm_units)
            if not self.return_attention_weights:
                out = self.attention(query, out, **kwargs)  # (batch_size, 1, n_time_steps, lstm_units)
            else:
                out, attention_weights = self.attention(query, out, return_attention_scores=True, **kwargs)
                attention_weights = tf.squeeze(attention_weights, axis=2)
            out = tf.squeeze(out, axis=1)  # Remove the extra dimension (batch_size, n_time_steps, lstm_units)
            out = self.pooling(out, **kwargs)  # (batch_size, 1, lstm_units)
        else:
            # pooling over groups, this totally invariants to the order of the groups
            out = self.pooling(out, **kwargs)  # (batch_size, lstm_units)
        # apply dense layer
        out = self.out_layer(out, **kwargs)  # (batch_size, summary_dim)

        if self.use_attention and self.return_attention_weights:
            return out, attention_weights
        return out

    def get_config(self):
        """Return the config for serialization."""
        config = super().get_config()
        config.update({
            'summary_dim': self.summary_dim,
            'num_conv_layers': self.num_conv_layers,
            'rnn_units': self.rnn_units,
            'bidirectional': self.bidirectional,
            'conv_settings': self.conv_settings,
            'use_attention': self.use_attention,
            'return_attention_weights': self.return_attention_weights,
            'use_GRU': self.use_GRU,
        })
        return config

    @classmethod
    def from_config(cls, config):
        """Recreate the model from the config."""
        return cls(**config)


def load_model(model_id: int,
               x_mean: np.ndarray, x_std: np.ndarray,
               p_mean: np.ndarray, p_std: np.ndarray,
               summary_valid_max: np.ndarray = None, summary_valid_min: np.ndarray = None,
               generative_model=None):
    # Set the logger to the desired level
    tf.get_logger().setLevel('ERROR')  # This will suppress warnings and info logs from TensorFlow

    n_params = 4
    num_coupling_layers = 6
    num_dense = 3
    use_attention = True
    use_bidirectional = True
    summary_loss = 'MMD'
    use_manual_summary = False
    if model_id == 0:
        checkpoint_path = 'amortizer-cell-migration-attention-6'
        map_idx_sim = 52
    elif model_id == 1:
        checkpoint_path = 'amortizer-cell-migration-attention-6-manual'
        use_manual_summary = True
        map_idx_sim = 6
    elif model_id == 2:
        checkpoint_path = 'amortizer-cell-migration-attention-7'
        num_coupling_layers = 7
        map_idx_sim = 52
    elif model_id == 3:
        checkpoint_path = 'amortizer-cell-migration-attention-7-manual'
        num_coupling_layers = 7
        use_manual_summary = True
        map_idx_sim = 28
    elif model_id == 4:
        checkpoint_path = 'amortizer-cell-migration-attention-8'
        num_coupling_layers = 8
        map_idx_sim = 69
    elif model_id == 5:
        checkpoint_path = 'amortizer-cell-migration-attention-8-manual'
        num_coupling_layers = 8
        use_manual_summary = True
        map_idx_sim = 86
    else:
        raise ValueError('Checkpoint path not found')

    summary_net = GroupSummaryNetwork(summary_dim=n_params * 2,
                                      rnn_units=32,
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

    amortizer = AmortizedPosterior(inference_net=inference_net, summary_net=summary_net,
                                   summary_loss_fun=summary_loss)

    # Disable logging
    logging.disable(logging.CRITICAL)

    # build the trainer with networks and generative model
    max_to_keep = 17
    trainer = Trainer(amortizer=amortizer,
                      configurator=partial(configurator,
                                           x_mean=x_mean, x_std=x_std,
                                           p_mean=p_mean, p_std=p_std,
                                           summary_valid_max=summary_valid_max, summary_valid_min=summary_valid_min,
                                           manual_summary=use_manual_summary),
                      generative_model=generative_model,
                      checkpoint_path=checkpoint_path,
                      skip_checks=True,  # simulation takes too much time
                      max_to_keep=max_to_keep)

    # check if file exist
    if os.path.exists(checkpoint_path):
        trainer.load_pretrained_network()
        history = trainer.loss_history.get_plottable()

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
        #print(f"Networks loaded from {new_checkpoint} with {recent_losses['Loss'][best_valid_epoch - 1]} validation loss")
    else:
        raise ValueError(f'Checkpoint path {checkpoint_path} not found')

    # Re-enable logging
    logging.disable(logging.NOTSET)

    return trainer, map_idx_sim


# get the job array id and number of processors
job_array_id = int(os.environ.get('SLURM_ARRAY_TASK_ID', 0))
n_procs = int(os.environ.get('SLURM_CPUS_PER_TASK', 1))
print(job_array_id)
on_cluster = True
population_size = 1000
run_old_sumstats = True

parser = argparse.ArgumentParser(description='Parse necessary arguments')
parser.add_argument('-pt', '--port', type=str, default="50004",
                    help='Which port should be use?')
parser.add_argument('-ip', '--ip', type=str,
                    help='Dynamically passed - BW: Login Node 3')
args = parser.parse_args()

#%%
if on_cluster:
    gp = '/home/jarruda_hpc/CellMigration/synth_data_params_bayesflow'
else:
    gp = os.getcwd()

par_map = {
    'gradient_strength': './CellTypes/CellType/Constant[@symbol="gradient_strength"]',
    'move.strength': './CellTypes/CellType/Constant[@symbol="move.strength"]',
    'move.duration.mean': './CellTypes/CellType/Constant[@symbol="move.duration.mean"]',
    'cell_nodes_real': './Global/Constant[@symbol="cell_nodes_real"]',
}


model_path = gp + "/cell_movement_v24.xml"  # time step is 30sec, move.dir completely normalized, init move.dir rand in all directions
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
    (ad_mean, _, msd_mean, _, ta_mean, _, vel_mean, _, wt_mean, _) = compute_mean_summary_stats([summary_stats_dict], remove_nan=False)
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


obs_pars_log = {key: np.log10(val) for key, val in obs_pars.items()}
limits = {'gradient_strength': (1, 10000), #(10 ** 4, 10 ** 8),
          'move.strength': (1, 100),
          'move.duration.mean': (1e-4, 30), #(math.log10((10 ** -2) * 30), math.log10((10 ** 4))), # smallest time step in simulation 5
          'cell_nodes_real': (1, 300)}
limits_log = {key: (np.log10(val[0]), np.log10(val[1])) for key, val in limits.items()}


prior = pyabc.Distribution(**{key: pyabc.RV("uniform", loc=lb, scale=ub-lb)
                              for key, (lb, ub) in limits_log.items()})
param_names = list(obs_pars.keys())
print(obs_pars)
#%%
# simulate test data
test_params = np.array(list(obs_pars_log.values()))
if not os.path.exists(os.path.join(gp, 'test_sim.npy')):
    raise FileNotFoundError('Test data not found')
else:
    test_sim = np.load(os.path.join(gp, 'test_sim.npy'))
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
    redis_sampler = RedisEvalParallelSampler(host=args.ip, port=args.port,
                                             adapt_look_ahead_proposal=False,
                                             look_ahead=False)

    abc = pyabc.ABCSMC(model, prior,
                       distance_function=obj_func_wass,
                       summary_statistics=make_sumstat_dict,
                       population_size=population_size,
                       sampler=redis_sampler)

    db_path = os.path.join(gp, "synthetic_test_old_sumstats.db")
    history = abc.new("sqlite:///" + db_path, make_sumstat_dict(test_sim))

    #start the abc fitting
    abc.run(min_acceptance_rate=1e-2, max_nr_populations=30)
    print('Done!')
    exit()

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
print('Mean and std of data:', x_mean, x_std)
print('Mean and std of parameters:', p_mean, p_std)

# compute the mean of the summary statistics
summary_stats_list_ = [reduced_coordinates_to_sumstat(t) for t in valid_data['sim_data']]
(_, ad_averg, _, MSD_averg, _, TA_averg, _, VEL_averg, _, WT_averg) = compute_mean_summary_stats(summary_stats_list_,
                                                                                                 remove_nan=False)

direct_conditions_ = np.stack([ad_averg, MSD_averg, TA_averg, VEL_averg, WT_averg]).T
# replace inf with -1
direct_conditions_[np.isinf(direct_conditions_)] = np.nan

summary_valid_max = np.nanmax(direct_conditions_, axis=0)
summary_valid_min = np.nanmin(direct_conditions_, axis=0)

#%%
# use trained neural net as summary statistics
def make_sumstat_dict_nn(
        data: Union[dict, np.ndarray],
) -> dict:
    if isinstance(data, dict):
        # get key
        key = list(data.keys())[0]
        data = data[key]

    trainer, map_idx_sim = load_model(
        model_id=5,
        x_mean=x_mean,
        x_std=x_std,
        p_mean=p_mean,
        p_std=p_std,
        summary_valid_max=summary_valid_max,
        summary_valid_min=summary_valid_min,
    )

    # configures the input for the network
    config_input = trainer.configurator({"sim_data": data})
    # get the summary statistics
    out_dict = {
        'summary_net': trainer.amortizer.summary_net(config_input['summary_conditions']).numpy().flatten()
    }
    # if direct conditions are available, concatenate them
    if 'direct_conditions' in config_input.keys():
        out_dict['direct_conditions'] = config_input['direct_conditions'].flatten()

    del trainer
    return out_dict


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
redis_sampler = RedisEvalParallelSampler(host=args.ip, port=args.port,
                                         adapt_look_ahead_proposal=False,
                                         look_ahead=False)
abc = pyabc.ABCSMC(model_nn, prior, # here we use now the Euclidean distance
                   population_size=population_size,
                   summary_statistics=make_sumstat_dict_nn,
                   sampler=redis_sampler)
db_path = os.path.join(gp, "synthetic_test_nn_sumstats.db")
history = abc.new("sqlite:///" + db_path, make_sumstat_dict_nn(test_sim))

#start the abc fitting
abc.run(min_acceptance_rate=1e-2, max_nr_populations=30)
#%%
print('Done!')
