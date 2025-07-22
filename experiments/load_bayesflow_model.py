import logging
import os
import pickle
from functools import partial

import keras
import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm

from bayesflow import default_settings as defaults
from bayesflow.amortizers import AmortizedPosterior
from bayesflow.helper_networks import MultiConv1D
from bayesflow.networks import InvertibleNetwork
from bayesflow.trainers import Trainer
from tensorflow.keras.layers import Dense, GRU, LSTM, Bidirectional
from tensorflow.keras.models import Sequential

on_cluster = False


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

        def log_posterior(self, configured_data: list[dict], n_samples: int = 1) -> np.ndarray:
            if self.n_amortizers != len(configured_data):
                raise ValueError(f'Number of summary_conditions ({len(configured_data)})'
                                 f' does not match number of amortizers ({self.n_amortizers}).')

            log_posteriors = []
            for _ in tqdm(range(n_samples)):
                log_posterior_a = []
                for a_i, amortizer in enumerate(self.amortizers):
                    configured_data[a_i].update({'parameters': amortizer.sample(configured_data[a_i], n_samples=1)})
                    log_posterior_a.append(amortizer.log_posterior(configured_data[a_i]))
                log_posteriors.append(log_posterior_a)
            return np.array(log_posteriors).reshape(self.n_amortizers, n_samples)

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
        print('Loading ensemble model')
        model_ids = [2, 1, 0]
        trainers = []
        for m_id in model_ids:
            trainer = load_model(m_id, x_mean, x_std, p_mean, p_std, generative_model)
            trainers.append(trainer)
        return EnsembleTrainer(trainers)
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
        #print(f"Networks loaded from {new_checkpoint} with {recent_losses['Loss'][best_valid_epoch - 1]} validation loss")

    # Re-enable logging
    logging.disable(logging.NOTSET)

    return trainer
