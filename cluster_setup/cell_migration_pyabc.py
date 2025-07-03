#%%
import argparse
import logging
import os
import pickle
from functools import partial
from typing import Optional, Union

import keras
import numpy as np
import pandas as pd
import math
import pyabc
import scipy.stats as stats
import tensorflow as tf
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

test_id = 0
run_manual_sumstats = True
use_npe_summaries = False
load_synthetic_data = True


# defining the summary statistics functions
def _turning_angle(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Compute the angle between two consecutive points."""
    vx = np.diff(x)
    vy = np.diff(y)
    theta = np.arctan2(vy, vx)
    theta = np.unwrap(theta)
    theta = np.diff(theta)
    return theta


def turning_angle(data_dict: dict) -> np.ndarray:
    """Compute the angle between two consecutive points, skipping segments with NaN."""
    x = np.array(data_dict['x'])
    y = np.array(data_dict['y'])

    # Find valid points (both x and y are not NaN)
    valid_mask = ~np.isnan(x) & ~np.isnan(y)

    # Initialize the list of turning angles
    turning_angles = []

    # Iterate through the valid mask to find continuous valid segments
    current_segment = []
    for i in range(len(valid_mask)):
        if valid_mask[i]:
            current_segment.append(i)
        else:
            # If we hit a NaN, process the current segment and reset
            if len(current_segment) > 1:
                segment_x = x[current_segment]
                segment_y = y[current_segment]
                # Compute turning angles for this segment
                ta = _turning_angle(segment_x, segment_y)
                turning_angles.extend(ta)
            current_segment = []

    # Process the last segment if it exists
    if len(current_segment) > 1:
        segment_x = x[current_segment]
        segment_y = y[current_segment]
        ta = _turning_angle(segment_x, segment_y)
        turning_angles.extend(ta)

    return np.array(turning_angles)


def velocity(data_dict: dict, dt: float = 30) -> np.ndarray:
    """Compute the velocity of the cell."""
    x = np.asarray(data_dict['x'])
    y = np.asarray(data_dict['y'])
    vx = np.diff(x)
    vy = np.diff(y)
    v = np.sqrt(vx ** 2 + vy ** 2) / dt
    return v


def compute_msd(
        trajectory: dict,
        all_time_lags: bool = True,
        dt: float = 30.0
) -> Union[np.ndarray, float]:
    """
    Compute the mean square displacement (MSD) on the interval where both x and y are observed.

    Parameters
    ----------
    trajectory : dict
        Must contain 1D arrays at keys x_key and y_key.
    all_time_lags : bool
        If True, return MSD for each lag from 1 up to the length of the observed window minus 1.
        If False, return only the maximal-lag MSD (first→last) divided by sqrt(total_time).
    dt : float
        Time step between successive frames.

    Returns
    -------
    np.ndarray or float
        - If all_time_lags=True: 1D array of length M-1 (where M is #observed points) with MSD at each lag.
        - If all_time_lags=False: single float = MSD(max_lag)/sqrt(max_lag * dt).
    """
    x = np.asarray(trajectory['x'], dtype=float)
    y = np.asarray(trajectory['y'], dtype=float)
    if x.shape != y.shape:
        raise ValueError("x and y must have the same length")

    # find the window where both x and y are non-NaN
    valid = (~np.isnan(x)) & (~np.isnan(y))
    idx = np.where(valid)[0]
    if idx.size < 2:
        # fewer than 2 valid points → no displacement
        return np.array([]) if all_time_lags else np.nan

    start, end = idx[0], idx[-1]
    x_obs = x[start:end + 1]
    y_obs = y[start:end + 1]
    M = len(x_obs)
    max_lag = M - 1

    # fast path: only the max-lag MSD from first→last
    if not all_time_lags:
        dx = x_obs[-1] - x_obs[0]
        dy = y_obs[-1] - y_obs[0]
        msd_end = dx * dx + dy * dy
        total_time = max_lag * dt
        return msd_end / np.sqrt(total_time)

    # full MSD curve
    msd = np.full(max_lag, np.nan, dtype=float)
    for lag in range(1, M):
        # displacements for this lag
        del_x = x_obs[lag:] - x_obs[:-lag]
        del_y = y_obs[lag:] - y_obs[:-lag]
        good = (~np.isnan(del_x)) & (~np.isnan(del_y))
        if good.any():
            msd[lag - 1] = np.mean(del_x[good] ** 2 + del_y[good] ** 2)

    return msd


def angle_degree(data_dict: dict) -> np.ndarray:
    """Compute the absolute angle between two consecutive points in degrees with respect to the x-axis."""
    x = data_dict['x']
    y = data_dict['y']
    vx = np.diff(x)
    vy = np.diff(y)
    list_angle_degrees = []
    for x, y in zip(vx, vy):
        list_angle_degrees.append(math.degrees(math.atan2(x, y)))
    return np.array(list_angle_degrees)


# my functions
def cut_region(
        data_dict: dict, x_min: float, x_max: float, y_min: float, y_max: float, return_longest: bool
) -> Optional[list[dict]]:
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


def compute_mean(x: Union[float, int, list, np.ndarray]) -> Union[float, np.floating]:
    """
    Compute the mean if x is a non-empty list.
    Return x if it's a float/int.
    Return np.nan if it's an empty list or unrecognized type.
    """
    if isinstance(x, (list, np.ndarray)):
        if len(x) == 0:
            return np.nan
        return np.nanmean(x)
    elif isinstance(x, (int, float)):
        return x
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


def compute_summary_stats(cell_population: np.ndarray, dt: float = 30) -> (list, list, list, list, list):
    """
    Compute the statistics of the reduced/visible coordinates of each cell in a cell population.

    :param cell_population: 3D array of cell populations
    :param dt: time interval between successive frames
    :return: list of msd, ta, v, ad
    """
    msd_list = []
    ta_list = []
    v_list = []
    ad_list = []

    cell_count = cell_population.shape[0]
    # check if sim_dict_cut is empty
    for i in range(0, cell_count):
        # select another cell
        sim_dict = {'x': cell_population[i, :, 0], 'y': cell_population[i, :, 1]}
        if all(np.isnan(sim_dict['x'])):
            continue
        else:
            msd_list.append(compute_mean(compute_msd(sim_dict, all_time_lags=False, dt=dt)))  # mean just for formatting
            ta_list.append(compute_mean(turning_angle(sim_dict)))
            v_list.append(compute_mean(velocity(sim_dict)))
            ad_list.append(compute_mean(angle_degree(sim_dict)))

    return remove_nan(msd_list), remove_nan(ta_list), remove_nan(v_list), remove_nan(ad_list)


def compute_MSD_lags(cell_population: np.ndarray) -> np.ndarray:
    """
    Compute the mean square displacement of the cell population for all possible time lags.

    :param cell_population: 3D array of cell populations
    :return: array of mean square displacements
    """
    msd_list = []
    cell_count = cell_population.shape[0]
    for i in range(0, cell_count):
        sim_dict = {'x': cell_population[i, :, 0], 'y': cell_population[i, :, 1]}
        if all(np.isnan(sim_dict['x'])):
            continue
        else:
            msd_list.append(compute_msd(sim_dict, all_time_lags=True))
    msd = np.stack(msd_list, axis=0)
    return msd


def reduce_to_coordinates(sumstat: dict,
                          minimal_length: int = 0,
                          maximal_length: Optional[int] = None,
                          only_longest_traj_per_cell: bool = True,
                          cut_region_of_interest: bool = True) -> list[dict]:
    """
    Reduce the output of the model to the visible coordinates of the cells.

    :param sumstat: output of Morpheus model
    :param minimal_length: minimal length of the trajectory
    :param maximal_length: maximal length of the trajectory
    :param only_longest_traj_per_cell: if True, only the longest trajectory of each cell is returned
    :param cut_region_of_interest: if True, the region of interest is cut
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
        if cut_region_of_interest:
            # cut the region of interest, divide cells into multiple cells if they leave the region
            sim_dict_cut = cut_region(sim_dict,
                                      x_min=316.5, x_max=856.5, y_min=1145, y_max=1351,
                                      return_longest=only_longest_traj_per_cell)
            # check if sim_dict_cut is empty
            if sim_dict_cut is not None:
                sim_list.extend(sim_dict_cut)
            else:
                continue
        else:
            # if no region of interest is cut, just append the cell
            sim_list.append(sim_dict)

    # only keep cells of minimal length
    if minimal_length > 0:
        sim_list = [sim for sim in sim_list if len(sim['x']) > minimal_length]
    if maximal_length is not None:
        sim_list = [{'x': sim['x'][:maximal_length],
                     'y': sim['y'][:maximal_length],
                     't': np.array(sim['t'][:maximal_length])
                     } for sim in sim_list]
    return sim_list


def remove_nan(stat_list: list):
    """
    Remove NaN and Inf from the list and compute the mean.
    """
    cleaned = [x for x in stat_list if not np.isnan(x) and not np.isinf(x)]
    return cleaned


def load_real_data(data_id: int, max_sequence_length: int, cells_in_population: int):

    if not data_id in [0, 1]:
        raise ValueError("The data_id should be either 0 or 1")

    # load real data
    real_data_id = [0, 1][data_id]
    if real_data_id == 0:
        real_data_df = pd.read_csv('/home/jarruda_hpc/CellMigration/real_data/37C-1_crop.csv')
        y_size = 739.79  # microns
        x_size = 279.74  # microns
        x_offset, y_offset = -7.5, -2  # correction such that the pillars are in the middle
        y_lin_shift = 0
    else:
        # more important data set, closer to the truth
        real_data_df = pd.read_csv('/home/jarruda_hpc/CellMigration/real_data/37C_ctrl2.csv')
        y_size = 882.94  # microns
        x_size = 287.03  # microns
        x_offset, y_offset = -2, -7.5  # 3, 7.5  # correction such that the pillars are in the middle
        y_lin_shift = -0.04  # 0.05  # correction for the tilt of the data

    # define the window
    # simulation coordinates, length of the gap: 1351-1145 = 206
    # in real data: 270
    # 270/206 = 1.31
    # morpheus size of image: 1173x2500
    factor = 1.31
    # reconstruction of the window assuming that it is centered
    # morpheus coordinates: 1173x2500x0
    window_x1, window_x2 = (1173 - y_size / factor) / 2., 1173 - (1173 - y_size / factor) / 2.
    window_y1, window_y2 = (2500 - x_size / factor) / 2., 2500 - (2500 - x_size / factor) / 2.

    # remove first three rows
    real_data_df = real_data_df.iloc[3:]
    # only keep positions and time
    real_data_df = real_data_df[['TRACK_ID', 'POSITION_X', 'POSITION_Y', 'POSITION_T']]
    # convert to numeric
    real_data_df = real_data_df.apply(pd.to_numeric, errors='coerce')

    # scale to morpheus coordinates
    real_data_scaled_df = real_data_df.copy()
    real_data_scaled_df['x'] = real_data_scaled_df['POSITION_X'] / factor + window_x1
    real_data_scaled_df['y'] = real_data_scaled_df['POSITION_Y'] / factor + window_y1
    # real cells are moving downwards, but in simulations they are going upwards
    real_data_scaled_df['y'] = 2500 - real_data_scaled_df['y']
    # data does not fit optimally, so we need to shift it
    real_data_scaled_df['x'] += x_offset
    real_data_scaled_df['y'] += y_offset
    # data is tilted
    real_data_scaled_df['x'] = real_data_scaled_df['x'] + (window_y2 - real_data_scaled_df['y']) * y_lin_shift
    # time
    real_data_scaled_df['t'] = real_data_scaled_df['POSITION_T']

    real_data = []
    cut_region_real_data = True
    sequence_lengths_real = []
    # each cell is of different length, each with x and y coordinates, make a tensor out of it
    for s_id, sample in enumerate(real_data_scaled_df.TRACK_ID.unique()):
        cell = real_data_scaled_df[real_data_scaled_df.TRACK_ID == sample]
        # order by time
        cell = cell.sort_values('POSITION_T', ascending=True)
        sequence_lengths_real.append(len(cell['y']))
        if cut_region_real_data:
            cell = cut_region(cell, x_min=316.5, x_max=856.5, y_min=1145, y_max=1351, return_longest=True)
            if cell is None:
                continue
            cell = cell[0]
        # pre-pad the data with zeros, but first write zeros as nans to compute the mean and std
        track = np.ones((max_sequence_length + 1, 3)) * np.nan
        track[-len(cell['x'][:max_sequence_length]):, 0] = cell['x'][:max_sequence_length]
        track[-len(cell['y'][:max_sequence_length]):, 1] = cell['y'][:max_sequence_length]
        track[-len(cell['t'][:max_sequence_length]):, 2] = np.round(cell['t'][:max_sequence_length])
        real_data.append(track[1:])  # remove the first time point, same as in the simulation (often nan anyway)

    real_data = np.stack(real_data)
    real_data_full = real_data.copy()
    real_data = real_data[:cells_in_population * (real_data_full.shape[0] // cells_in_population)]
    return real_data, real_data_full


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


def configurator(
        forward_dict: dict,
        x_mean: np.ndarray, x_std: np.ndarray,
        p_mean: np.ndarray, p_std: np.ndarray,
) -> dict:
    out_dict = {}

    # Extract data
    x = forward_dict["sim_data"]

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



class SummaryNetwork(tf.keras.Model):
    """Network to summarize the population of cells."""

    def __init__(
            self,
            summary_dim,
            num_conv_layers=2,
            rnn_units=128,
            bidirectional=True,
            conv_settings=None,
            use_GRU=True,
            **kwargs
    ):
        super().__init__(**kwargs)

        if conv_settings is None:
            conv_settings = defaults.DEFAULT_SETTING_MULTI_CONV
        self.conv_settings = conv_settings

        self.conv = Sequential([MultiConv1D(conv_settings) for _ in range(num_conv_layers)])
        self.num_conv_layers = num_conv_layers
        self.rnn_units = rnn_units
        self.use_GRU = use_GRU
        self.bidirectional = bidirectional


        if use_GRU:
            self.rnn = Bidirectional(GRU(rnn_units)) if bidirectional else GRU(rnn_units)
        else:
            self.rnn = Bidirectional(LSTM(rnn_units)) if bidirectional else LSTM(rnn_units)

        self.out_layer = Dense(summary_dim, activation="linear")
        self.summary_dim = summary_dim

    def call(self, x, **kwargs):
        """Performs a forward pass through the network by first passing `x` through the rnn network.

        Parameters
        ----------
        x : tf.Tensor
            Input of shape (batch_size, n_groups, n_time_steps, n_features)

        Returns
        -------
        out : tf.Tensor
            Output of shape (batch_size, summary_dim)
        """
        # transform (batch_size, n_groups, n_time_steps, n_features) to (batch_size, n_time_steps, n_groups*n_features)
        out = tf.transpose(x, [0, 2, 1, 3])  # transpose to (batch_size, n_time_steps, n_groups, n_features)
        out = tf.reshape(out, [-1, out.shape[1], out.shape[2] * out.shape[3]])

        # Apply the RNN
        out = self.conv(out, **kwargs)
        out = self.rnn(out, **kwargs)  # (batch_size, lstm_units)
        # bidirectional LSTM returns 2*lstm_units

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
            'bidirectional': self.bidirectional,
            'conv_settings': self.conv_settings,
            'use_GRU': self.use_GRU,
        })
        return config

    @classmethod
    def from_config(cls, config):
        """Recreate the model from the config."""
        return cls(**config)


# define the network
class GroupSummaryNetwork(tf.keras.Model):
    """Network to summarize the data of groups of cells.  Each group is passed through a series of convolutional layers
    followed by a GRU layer. The output of the GRU layer is then pooled across the groups and dense layer applied
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
        each cell and then pooling the outputs across cells.

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


class EnsembleTrainer:
    def __init__(self, trainers):
        self.trainers = trainers
        self.n_trainers = len(trainers)
        self.checkpoint_path = 'amortizer-cell-migration-ensemble'
        self.amortizer = self.EnsembleAmortizer([trainer.amortizer for trainer in trainers])
        self.loss_history = self.EnsembleLossHistory(trainers)

    def configurator(self, forward_dict: dict) -> list[dict]:
        out_list = []
        for trainer in self.trainers:
            out = trainer.configurator(forward_dict)
            out_list.append(out)
        return out_list

    class EnsembleAmortizer:
        def __init__(self, amortizers):
            self.amortizers = amortizers
            self.n_amortizers = len(amortizers)
            self.summary_loss = None

        def sample(self, forward_dict: list[dict], n_samples: int) -> np.ndarray:
            if self.n_amortizers != len(forward_dict):
                raise ValueError(f'Number of forward_dicts ({len(forward_dict)})'
                                 f' does not match number of amortizers ({self.n_amortizers}).')

            out_list = []
            n_samples_per_amortizer = np.ones(self.n_amortizers) * (n_samples // self.n_amortizers)
            n_samples_per_amortizer[:n_samples % self.n_amortizers] += 1

            for a_i, amortizer in enumerate(self.amortizers):
                out = amortizer.sample(forward_dict[a_i], n_samples=n_samples_per_amortizer[a_i])
                out_list.append(out)
            if out_list[0].ndim == 2:
                return np.concatenate(out_list, axis=0)
            return np.concatenate(out_list, axis=1)

        def summary_net(self, summary_conditions: np.ndarray) -> np.ndarray:
            if self.n_amortizers != len(summary_conditions):
                raise ValueError(f'Number of summary_conditions ({len(summary_conditions)})'
                                 f' does not match number of amortizers ({self.n_amortizers}).')

            out_list = []
            for a_i, amortizer in enumerate(self.amortizers):
                out = amortizer.summary_net(summary_conditions[a_i]['summary_conditions'])
                out_list.append(out)
            return np.concatenate(out_list, axis=1)

    class EnsembleLossHistory:
        def __init__(self, trainers):
            self.trainers = trainers

        def get_plottable(self):
            # Collect all DataFrames for each trainer's train and validation losses
            train_dfs = []
            val_dfs = []

            for trainer in self.trainers:
                history = trainer.loss_history.get_plottable()
                train_dfs.append(history['train_losses'])
                val_dfs.append(history['val_losses'])

            # Calculate the average DataFrame across trainers for both train and val losses
            avg_train_df = pd.concat(train_dfs).groupby(level=0).mean()
            avg_val_df = pd.concat(val_dfs).groupby(level=0).mean()

            return {
                'train_losses': avg_train_df,
                'val_losses': avg_val_df
            }


def load_model(model_id: int,
               x_mean: np.ndarray, x_std: np.ndarray,
               p_mean: np.ndarray, p_std: np.ndarray,
               generative_model=None):
    # Set the logger to the desired level
    tf.get_logger().setLevel('ERROR')  # This will suppress warnings and info logs from TensorFlow

    n_params = 4
    num_coupling_layers = 6
    num_dense = 3
    use_attention = True
    use_bidirectional = True
    summary_loss = 'MMD'
    summary_net = None  # will be defined later
    if model_id == 0:
        checkpoint_path = 'amortizer-cell-migration-6'
    elif model_id == 1:
        checkpoint_path = 'amortizer-cell-migration-7'
        num_coupling_layers = 7
    elif model_id == 2:
        checkpoint_path = 'amortizer-cell-migration-8'
        num_coupling_layers = 8
    elif model_id == 3:
        raise 'Checkpoint path not found'
    elif model_id == 10:
        print('load only summary model without checkpoint')
        checkpoint_path = 'amortizer-only-summary'
        num_coupling_layers = 1
        summary_net = SummaryNetwork(
            summary_dim=n_params,
            rnn_units=32,
            bidirectional=use_bidirectional
        )
    else:
        raise ValueError('Checkpoint path not found')

    if on_cluster:
        checkpoint_path = "/home/jarruda_hpc/CellMigration/synth_data_params_bayesflow/" + checkpoint_path

    if summary_net is None:
        summary_net = GroupSummaryNetwork(
            summary_dim=n_params * 2,
            rnn_units=32,
            use_attention=use_attention,
            bidirectional=use_bidirectional
        )

    inference_net = InvertibleNetwork(
        num_params=n_params,
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
        }
    )

    amortizer = AmortizedPosterior(
        inference_net=inference_net,
        summary_net=summary_net,
        summary_loss_fun=summary_loss
    )

    # Disable logging
    logging.disable(logging.CRITICAL)

    # build the trainer with networks and generative model
    max_to_keep = 17
    trainer = Trainer(
        amortizer=amortizer,
        configurator=partial(configurator,
                             x_mean=x_mean, x_std=x_std,
                             p_mean=p_mean, p_std=p_std),
        generative_model=generative_model,
        checkpoint_path=checkpoint_path,
        skip_checks=True,  # simulation takes too much time
        max_to_keep=max_to_keep
    )

    # check if file exist
    if os.path.exists(checkpoint_path):
        if model_id == 10:
            # keras model not BayesFlow
            trainer.amortizer.summary_net = keras.models.load_model(trainer.checkpoint_path,
                                                                    custom_objects={'summary_net': summary_net})
            # Re-enable logging
            logging.disable(logging.NOTSET)

            return trainer

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
        # print(f"Networks loaded from {new_checkpoint} with {recent_losses['Loss'][best_valid_epoch - 1]} validation loss")

    # Re-enable logging
    logging.disable(logging.NOTSET)

    return trainer


# get the job array id and number of processors
n_procs = int(os.environ.get('SLURM_CPUS_PER_TASK', 1))
print('test_id', test_id)
on_cluster = True
population_size = 1000

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
    msd_list, ta_list, v_list, ad_list = compute_summary_stats(data)
    cleaned_dict = {
        'msd': np.array(msd_list).flatten(),
        'ta': np.array(ta_list).flatten(),
        'vel': np.array(v_list).flatten(),
        'ad': np.array(ad_list).flatten(),
    }
    return cleaned_dict


def prepare_sumstats(output_morpheus_model) -> dict:
    sim_coordinates = reduce_to_coordinates(output_morpheus_model,
                                            minimal_length=min_sequence_length,
                                            maximal_length=max_sequence_length,
                                            only_longest_traj_per_cell=only_longest_traj_per_cell
                                            )

    # we now do exactly the same as in the BayesFlow workflow, but here we get only one sample at a time
    data_transformed = np.ones((1, cells_in_population, max_sequence_length, 3)) * np.nan
    # each cell is of different length, each with x and y coordinates, make a tensor out of it
    n_cells_not_visible = 0
    if len(sim_coordinates) != 0:
        # some cells were visible in the simulation
        for c_id, cell_sim in enumerate(sim_coordinates):
            # pre-pad the data with zeros, but first write zeros as nans to compute the mean and std
            data_transformed[0, c_id, -len(cell_sim['x']):, 0] = cell_sim['x']
            data_transformed[0, c_id, -len(cell_sim['y']):, 1] = cell_sim['y']
            data_transformed[0, c_id, -len(cell_sim['t']):, 2] = cell_sim['t']

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

    # note: remember also change tiff path in model.xml!
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
param_names = ['$m_{\\text{dir}}$', '$m_{\\text{rand}}$', '$w$', '$a$']
log_param_names = ['$\log_{10}(m_{\\text{dir}})$', '$\log_{10}(m_{\\text{rand}})$', '$\log_{10}(w)$', '$\log_{10}(a)$']
print(obs_pars)
#%%
if load_synthetic_data:
    # simulate test data
    if not os.path.exists(os.path.join(gp, f'test_sim_{test_id}.npy')):
        raise FileNotFoundError('Test data not found')
    else:
        test_sim = np.load(os.path.join(gp, f'test_sim_{test_id}.npy'))

    results_path = f'abc_results_{test_id}'
else:
    # load real data in morpheus format
    real_data, real_data_full = load_real_data(data_id=1,
                                               max_sequence_length=max_sequence_length,
                                               cells_in_population=cells_in_population)
    test_sim = real_data_full[np.newaxis]

#%%
def obj_func_wass_helper(sim: dict, obs: dict, key: str) -> float:
    x, y = np.array(sim[key]), np.array(obs[key])
    if x.size == 0 or y.size == 0:
        return np.inf
    return stats.wasserstein_distance(x, y)

distances = {
    'ad': pyabc.distance.FunctionDistance(partial(obj_func_wass_helper, key='ad')),
    'msd': pyabc.distance.FunctionDistance(partial(obj_func_wass_helper, key='msd')),
    'ta': pyabc.distance.FunctionDistance(partial(obj_func_wass_helper, key='ta')),
    'vel': pyabc.distance.FunctionDistance(partial(obj_func_wass_helper, key='vel')),
}
adaptive_weights = {
    d: 1. / max(np.max(make_sumstat_dict(test_sim)[d]), 1e-4) for d in distances.keys()
}

# adaptive distance
adaptive_wasserstein_distance = pyabc.distance.AdaptiveAggregatedDistance(
    distances=list(distances.values()),
    initial_weights=list(adaptive_weights.values()),
    adaptive=False,
    log_file=os.path.join(gp, f"adaptive_distance_log_{test_id}.txt")
)
#%%
if run_manual_sumstats:
    redis_sampler = RedisEvalParallelSampler(host=args.ip, port=args.port,
                                             adapt_look_ahead_proposal=False,
                                             look_ahead=False)

    abc = pyabc.ABCSMC(model, prior,
                       distance_function=adaptive_wasserstein_distance,
                       summary_statistics=make_sumstat_dict,
                       population_size=population_size,
                       sampler=redis_sampler)

    db_path = os.path.join(gp, f"{'synthetic_'+str(test_id) if load_synthetic_data else 'real'}_test_wasserstein_sumstats_adaptive.db")
    history = abc.new("sqlite:///" + db_path, make_sumstat_dict(test_sim))

    #start the abc fitting
    abc.run(min_acceptance_rate=1e-2, max_nr_populations=15)
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
#%%
# use trained neural net as summary statistics
def make_sumstat_dict_nn(data: Union[dict, np.ndarray], use_npe_summaries: bool = True) -> dict:
    if use_npe_summaries:
        model_id = 2
    else:
        model_id = 10
    if isinstance(data, dict):
        # get key
        key = list(data.keys())[0]
        data = data[key]

    trainer = load_model(
        model_id=model_id,
        x_mean=x_mean,
        x_std=x_std,
        p_mean=p_mean,
        p_std=p_std,
    )

    # configures the input for the network
    config_input = trainer.configurator({"sim_data": data})
    # get the summary statistics
    if isinstance(trainer, EnsembleTrainer):
        out_dict = {
            'summary_net': trainer.amortizer.summary_net(config_input).flatten()
        }
    else:
        out_dict = {
            'summary_net': trainer.amortizer.summary_net(config_input['summary_conditions']).numpy().flatten()
        }
    if model_id == 10:
        # renormalize the parameters
        out_dict['summary_net'] = out_dict['summary_net'] * p_std + p_mean

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

    # note: remember also change tiff path in model.xml!
else:
    # define the model object
    model_nn = morpheus_model.MorpheusModel(
        model_path, par_map=par_map, par_scale="log10",
        show_stdout=False, show_stderr=False,
        clean_simulation=True,
        raise_on_error=False, sumstat=sumstat)

#%%
obs_nn = make_sumstat_dict_nn(test_sim, use_npe_summaries=use_npe_summaries)
redis_sampler = RedisEvalParallelSampler(host=args.ip, port=args.port,
                                         adapt_look_ahead_proposal=False,
                                         look_ahead=False)
abc_nn = pyabc.ABCSMC(model_nn, prior, # here we use now the Euclidean distance
                   population_size=population_size,
                   summary_statistics=partial(make_sumstat_dict_nn, use_npe_summaries=use_npe_summaries),
                   sampler=redis_sampler)
if use_npe_summaries:
    db_path = os.path.join(gp,
                           f"{'synthetic_'+str(test_id) if load_synthetic_data else 'real'}_test_nn_sumstats.db")
else:
    db_path = os.path.join(gp,
                           f"{'synthetic_' + str(test_id) if load_synthetic_data else 'real'}_test_nn_sumstats_posterior_mean.db")
print(db_path)
history = abc_nn.new("sqlite:///" + db_path, obs_nn)

#start the abc fitting
abc_nn.run(min_acceptance_rate=1e-2, max_nr_populations=15)
#%%
print('Done!')
