import logging
import os
import pickle
from functools import partial

import numpy as np
import tensorflow as tf
from bayesflow import default_settings as defaults
from bayesflow.amortizers import AmortizedPosterior
from bayesflow.helper_networks import MultiConv1D
from bayesflow.networks import InvertibleNetwork
from bayesflow.trainers import Trainer
from tensorflow.keras.layers import Dense, GRU, LSTM, Bidirectional
from tensorflow.keras.models import Sequential

from load_data import load_real_data
from summary_stats import reduced_coordinates_to_sumstat, compute_mean_summary_stats

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
        summary_valid_min: np.ndarray = None, summary_valid_max: np.ndarray = None,
        manual_summary: bool = False,
        include_real: str = None
) -> dict:
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

    if include_real is not None:
        assert manual_summary == True
        max_sequence_length = 120
        cells_in_population = 50
        p = 1  # Lp-distance

        real_data, _ = load_real_data(
            data_id=1,
            max_sequence_length=max_sequence_length,
            cells_in_population=cells_in_population
        )
        real_data = np.array(
            [real_data[start:start + cells_in_population]
             for start in range(0, len(real_data), cells_in_population)])[0][np.newaxis]

        trainer, _ = load_model(
            model_id=5,
            x_mean=x_mean,
            x_std=x_std,
            p_mean=p_mean,
            p_std=p_std,
            summary_valid_max=summary_valid_max,
            summary_valid_min=summary_valid_min,
        )

        # configures the input for the network
        config_input = trainer.configurator({"sim_data": real_data})
        # get the summary statistics
        real_dict = {
            'summary_net': trainer.amortizer.summary_net(config_input['summary_conditions']).numpy(),
            'direct_conditions': config_input['direct_conditions']
        }

        # get the summary statistics of the batch
        batch_dict = {
            'summary_net': trainer.amortizer.summary_net(out_dict['summary_conditions']).numpy(),
            'direct_conditions': out_dict['direct_conditions']
        }
        real = np.concatenate((real_dict['summary_net'], real_dict['direct_conditions']), axis=-1)
        batch = np.concatenate((batch_dict['summary_net'], batch_dict['direct_conditions']), axis=-1)

        if include_real == 'concat':
            # concatenate difference to real data to the summary conditions
            out_direct = (np.abs(batch - real)**p).sum(axis=-1) ** (1/p)
            out_dict['direct_conditions'] = np.concatenate((
                batch_dict['summary_net'],  # no summary is trained, so we use the precomputed network
                out_dict['direct_conditions'],  # direct conditions as they are
                out_direct[:, np.newaxis]  # difference to real data (same norm as in pyABC)
            ), axis=-1).astype(np.float32)
        elif include_real == 'compare':
            # summary statistics is only relative to real data
            out_direct = (np.abs(batch - real)**p)
            out_dict['direct_conditions'] = out_direct.astype(np.float32)  # difference to real data (same norm as in pyABC, but not summed)
        else:
            raise ValueError('Include real type not recognized')
        del trainer
        del real_data

        # drop summary conditions
        out_dict.pop('summary_conditions')

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


class EnsembleAmortizer:
    def __init__(self, amortizers):
        self.amortizers = amortizers
        self.n_amortizers = len(amortizers)

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


class EnsembleTrainer:
    def __init__(self, trainers):
        self.trainers = trainers
        self.n_trainers = len(trainers)
        self.checkpoint_path = None
        self.loss_history = None
        self.amortizer = EnsembleAmortizer([trainer.amortizer for trainer in trainers])

    def configurator(self, forward_dict: dict) -> list[dict]:
        out_list = []
        for trainer in self.trainers:
            out = trainer.configurator(forward_dict)
            out_list.append(out)
        return out_list



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
    include_real = None
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
    elif model_id == 6:
        checkpoint_path = 'amortizer-cell-migration-attention-8-manual-include-real'
        num_coupling_layers = 8
        use_manual_summary = True
        map_idx_sim = np.nan
        include_real = 'concat'
        summary_loss = None
    elif model_id == 7:
        checkpoint_path = 'amortizer-cell-migration-attention-8-manual-compare-real'
        num_coupling_layers = 8
        use_manual_summary = True
        map_idx_sim = np.nan
        include_real = 'compare'
        summary_loss = None
    elif model_id == 8:
        print('Loading ensemble model')
        model_ids = [0, 1, 2, 3, 4, 5]
        trainers = []
        for m_id in model_ids:
            trainer, _ = load_model(m_id, x_mean, x_std, p_mean, p_std,
                                 summary_valid_max, summary_valid_min, generative_model)
            trainers.append(trainer)
        trainers = EnsembleTrainer(trainers)
        return trainers, np.nan  # no map_idx_sim for ensemble model
    else:
        raise ValueError('Checkpoint path not found')

    if on_cluster:
        checkpoint_path = "/home/jarruda_hpc/CellMigration/synth_data_params_bayesflow/" + checkpoint_path

    if include_real is None:
        summary_net = GroupSummaryNetwork(
            summary_dim=n_params * 2,
            rnn_units=32,
            use_attention=use_attention,
            bidirectional=use_bidirectional
        )
    else:
        summary_net = None

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
                             p_mean=p_mean, p_std=p_std,
                             summary_valid_max=summary_valid_max, summary_valid_min=summary_valid_min,
                             manual_summary=use_manual_summary, include_real=include_real),
        generative_model=generative_model,
        checkpoint_path=checkpoint_path,
        skip_checks=True,  # simulation takes too much time
        max_to_keep=max_to_keep
    )

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

    # Re-enable logging
    logging.disable(logging.NOTSET)

    return trainer, map_idx_sim
