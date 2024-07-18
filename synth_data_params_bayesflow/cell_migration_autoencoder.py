import numpy as np
import pandas as pd
import os
import tidynamics  # to get sliding history stats in N*logN instead of N^2
from fitmulticell import model as morpheus_model
import pyabc
import argparse
import math
from fitmulticell.sumstat import SummaryStatistics
import scipy.stats as stats
from tqdm import tqdm
import pickle
from typing import Optional

# handling the redis parameters
#parser = argparse.ArgumentParser(description='Parse necessary arguments')
#parser.add_argument('-pt', '--port', type=str, default="50004",
#                    help='Which port should be use?')
#parser.add_argument('-ip', '--ip', type=str,
#                    help='Dynamically passed - BW: Login Node 3')
#args = parser.parse_args()


# defining the summary statistics functions
def turning_angle(data_dict):
    x = data_dict['x']
    y = data_dict['y']
    vx = np.diff(x)
    vy = np.diff(y)
    theta = np.arctan2(vy, vx)
    theta = np.unwrap(theta)
    theta = np.diff(theta)
    return theta


def velocity(data_dict):
    x = data_dict['x']
    y = data_dict['y']
    vx = np.diff(x)
    vy = np.diff(y)
    v = np.sqrt(vx ** 2 + vy ** 2)
    return v


def MSD(data_dict, x_name="x", y_name="y"):
    msd = tidynamics.msd(
        np.column_stack([data_dict[x_name], data_dict[y_name]]))
    return msd


def angle_degree(data_dict):
    x = data_dict['x']
    y = data_dict['y']
    vx = np.diff(x)
    vy = np.diff(y)
    list_angle_degrees = []
    for x, y in zip(vx, vy):
        list_angle_degrees.append(math.degrees(math.atan2(x, y)))
    return list_angle_degrees


def read_data_from_csv_to_df(loc, filename):
    data_file = os.path.join(loc, filename)
    data_frame = pd.read_csv(data_file, sep="\t", encoding='unicode_escape')
    return data_frame


def save_XY_xolums_in_dict(datarame):
    x = np.array(datarame['X'])
    y = np.array(datarame['Y'])
    return {'x': x, 'y': y}


# take a np array and return a every 10th element starting from 3ed element
def get_xy_cell(array, n_cells, cell_id):
    return array[cell_id::n_cells]


def get_sumstat(sim_dict, sumstat, sumstat_name, cell_id=0, n_cells=10):
    new_dict = sim_dict[sumstat]
    cell_dict = get_xy_cell(new_dict, n_cells, cell_id)
    return {sumstat_name: cell_dict}


def cut_region(data_dict, x_min, x_max, y_min, y_max):
    x = data_dict['x']
    y = data_dict['y']
    x_cut = []
    y_cut = []
    for x_, y_ in zip(x, y):
        if x_min < x_ < x_max and y_min < y_ < y_max:
            x_cut.append(x_)
            y_cut.append(y_)
    return {'x': x_cut, 'y': y_cut}


# the main sumstat function that is calle after the model excution from Morpheus

gp = os.getcwd()
#gp = '/home/jarruda_hpc/CellMigration/synth_data_params_4_wass_var_50cell'


def coordinate_to_sumstat(sumstat):
    # get unique values of sumstat['cell.id']
    cell_count = len(np.unique(sumstat['cell.id']))
    # read csv file
    x_pos = get_sumstat(sumstat, "cell.center.x", "x", cell_id=0)
    y_pos = get_sumstat(sumstat, "cell.center.y", "y", cell_id=0)
    # remove first element of x and y
    x_pos['x'] = x_pos['x'][1:]
    y_pos['y'] = y_pos['y'][1:]
    sim_dict = {**x_pos, **y_pos}
    sim_dict_cut = cut_region(sim_dict, 316.5, 856.5, 1145, 1351)
    # check if sim_dict_cut is empty
    if not sim_dict_cut['x']:
        for i in range(1, cell_count):
            # select another cell
            x_pos = get_sumstat(sumstat, "cell.center.x", "x", cell_id=1)
            y_pos = get_sumstat(sumstat, "cell.center.y", "y", cell_id=1)
            # remove first element of x and y
            x_pos['x'] = x_pos['x'][1:]
            y_pos['y'] = y_pos['y'][1:]
            sim_dict = {**x_pos, **y_pos}
            sim_dict_cut = cut_region(sim_dict, 316.5, 856.5, 1145, 1351)
            # check if sim_dict_cut is empty
            if not sim_dict_cut['x']:
                continue
            else:
                break
        return {'msd': [np.inf], 'ta': [np.inf], 'velocity': [np.inf],
                'ad': [np.inf]}

    sim_msd = MSD(sim_dict_cut)
    sim_ta = turning_angle(sim_dict_cut)
    sim_velocity = velocity(sim_dict_cut)
    sim_ad = angle_degree(sim_dict_cut)
    sim = {'msd': sim_msd, 'ta': sim_ta, 'velocity': sim_velocity,
           'ad': sim_ad}
    return sim


def coordinate_to_sumstat_v2(sumstat):
    msd_list = []
    ta_list = []
    v_list = []
    ad_list = []

    # get unique values of sumstat['cell.id']
    cell_count = len(np.unique(sumstat['cell.id']))
    # read csv file
    # x_pos = get_sumstat(sumstat, "cell.center.x", "x", cell_id=0)
    # y_pos = get_sumstat(sumstat, "cell.center.y", "y", cell_id=0)
    # # remove first element of x and y
    # x_pos['x'] = x_pos['x'][1:]
    # y_pos['y'] = y_pos['y'][1:]
    # sim_dict = {**x_pos, **y_pos}
    # sim_dict_cut = cut_region(sim_dict, 316.5, 856.5, 1145, 1351)
    # check if sim_dict_cut is empty
    for i in range(0, cell_count):
        # select another cell
        x_pos = get_sumstat(sumstat, "cell.center.x", "x", cell_id=i)
        y_pos = get_sumstat(sumstat, "cell.center.y", "y", cell_id=i)
        # remove first element of x and y
        x_pos['x'] = x_pos['x'][1:]
        y_pos['y'] = y_pos['y'][1:]
        sim_dict = {**x_pos, **y_pos}
        sim_dict_cut = cut_region(sim_dict, 316.5, 856.5, 1145, 1351)
        # check if sim_dict_cut is empty
        if not sim_dict_cut['x']:
            continue
        else:
            msd_list.append(MSD(sim_dict_cut))
            ta_list.append(turning_angle(sim_dict_cut))
            v_list.append(velocity(sim_dict_cut))
            ad_list.append(angle_degree(sim_dict_cut))
    if not msd_list:
        return {'msd_mean': [np.inf], 'msd_var': [np.inf], 'ta_mean': [np.inf], 'ta_var': [np.inf],
                'v_mean': [np.inf], 'v_var': [np.inf], 'ad_mean': [np.inf], 'ad_var': [np.inf]}

    # get the mean of list of lists with different lengths
    msd_mean = [np.mean(x) for x in msd_list]
    ta_mean = [np.mean(x) for x in ta_list]
    v_mean = [np.mean(x) for x in v_list]
    ad_mean = [np.mean(x) for x in ad_list]

    # get the variance of list of lists with different lengths
    msd_var = [np.var(x) for x in msd_list]
    ta_var = [np.var(x) for x in ta_list]
    v_var = [np.var(x) for x in v_list]
    ad_var = [np.var(x) for x in ad_list]

    sim = {'msd_mean': msd_mean, 'msd_var': msd_var, 'ta_mean': ta_mean, 'ta_var': ta_var,
           'v_mean': v_mean, 'v_var': v_var, 'ad_mean': ad_mean, 'ad_var': ad_var}
    return sim


def coordinate_to_sumstat_v3(sumstat):
    sim_list = []

    # get unique values of sumstat['cell.id']
    cell_count = len(np.unique(sumstat['cell.id']))
    # check if sim_dict_cut is empty
    for i in range(0, cell_count):
        # select another cell
        x_pos = get_sumstat(sumstat, "cell.center.x", "x", cell_id=i)
        y_pos = get_sumstat(sumstat, "cell.center.y", "y", cell_id=i)
        # remove first element of x and y
        x_pos['x'] = x_pos['x'][1:]
        y_pos['y'] = y_pos['y'][1:]
        sim_dict = {**x_pos, **y_pos}
        sim_dict_cut = cut_region(sim_dict, 316.5, 856.5, 1145, 1351)
        # check if sim_dict_cut is empty
        if not sim_dict_cut['x']:
            continue
        else:
            sim_list.append(sim_dict_cut)

    if not sim_list:
        return {'x': [np.inf], 'y': [np.inf]}
    return sim_list


# defining the objective functions
def obj_func(sim, obs):
    total = 0
    for key in sim:
        if key == 'loc': continue
        x = np.array(sim[key])
        y = np.array(obs[key])
        if np.max(y) != 0:
            x = x / np.max(y)
            y = y / np.max(y)
        total += np.sum((x - y) ** 2) / x.size
    return total


def obj_func_zeros(sim, obs):
    total = 0
    # check the length of sim and obs
    # make a deep copy of sim and obs
    sim_c = sim.copy()
    obs_c = obs.copy()

    if len(sim['msd']) < len(obs['msd']):
        # add 0 to make the length of sim and obs equal
        sim_c['msd'] = np.append(sim['msd'], np.zeros(len(obs['msd']) - len(sim['msd'])))
        sim_c['ta'] = np.append(sim['ta'], np.zeros(len(obs['ta']) - len(sim['ta'])))
        sim_c['velocity'] = np.append(sim['velocity'], np.zeros(len(obs['velocity']) - len(sim['velocity'])))
        sim_c['ad'] = np.append(sim['ad'], np.zeros(len(obs['ad']) - len(sim['ad'])))
    elif len(sim['msd']) > len(obs['msd']):
        # add 0 to make the length of sim and obs equal
        obs_c['msd'] = np.append(obs['msd'], np.zeros(len(sim['msd']) - len(obs['msd'])))
        obs_c['ta'] = np.append(obs['ta'], np.zeros(len(sim['ta']) - len(obs['ta'])))
        obs_c['velocity'] = np.append(obs['velocity'], np.zeros(len(sim['velocity']) - len(obs['velocity'])))
        obs_c['ad'] = np.append(obs['ad'], np.zeros(len(sim['ad']) - len(obs['ad'])))

    for key in sim_c:
        if key == 'loc': continue
        x = np.array(sim_c[key])
        y = np.array(obs_c[key])
        if np.max(y) != 0:
            x = x / np.max(y)
            y = y / np.max(y)
        total += np.sum((x - y) ** 2) / x.size
    return total


# this is the current used one, all above objective functions are old ones
def obj_func_wass(sim, obs):
    total = 0
    for key in sim:
        if key == 'loc': continue
        x = np.array(sim[key])
        y = np.array(obs[key])
        if x.size == 0:
            return np.inf
        stats.wasserstein_distance(x, y)
        total += stats.wasserstein_distance(x, y)
    return total


# defining the mapping of parameter inside the model xml file. the dictionary name is for 
# parameter name, and the value are the mapping values, to get the map value for parameter 
# check here: https://fitmulticell.readthedocs.io/en/latest/example/minimal.html#Inference-problem-definition

par_map = {
    'gradient_strength': './CellTypes/CellType/Constant[@symbol="gradient_strength"]',
    'move.strength': './CellTypes/CellType/Constant[@symbol="move.strength"]',
    'move.duration.median': './CellTypes/CellType/Constant[@symbol="move.duration.median"]',
    #    'move.duration.scale': './CellTypes/CellType/Constant[@symbol="move.duration.scale"]',
    'cell_nodes': './Global/Constant[@symbol="cell_nodes"]',
    'd_env': './CellTypes/CellType/Property[@symbol="d_env"]',
}

model_path = gp + "/cell_movement_v21.xml"
# defining the summary statistics function
sumstat = SummaryStatistics(sum_stat_calculator=coordinate_to_sumstat_v3)  # coordinate_to_sumstat_v2)

# define the model object
model = morpheus_model.MorpheusModel(
    model_path, par_map=par_map, par_scale="lin",
    show_stdout=False, show_stderr=False,
    # TODO: my path
    #executable="ulimit -s unlimited; /home/jarruda_hpc/CellMigration/morpheus-2.3.7",
    raise_on_error=False, sumstat=sumstat)

# todo: also change tiff path in model.xml!
# todo: check redis sampler

# parameter values used to generate the synthetic data
obs_pars = {
    'gradient_strength': 500000,
    'move.strength': 0.0021,
    'move.duration.median': 972.33,
    # 'move.duration.scale': 96899.44,
    'cell_nodes': 30,
    'd_env': 0.01,
}

#calling the model to generate synthitic data
#synthetic_data = model.sample(obs_pars)

# define the parameter scale
model.par_scale = "log10"
# define parameters' limits
obs_pars_log = {key: math.log10(val) for key, val in obs_pars.items()}
limits = {key: (math.log10((10 ** -3) * val), math.log10((10 ** 3) * val)) for
          key, val in obs_pars.items()}

limits['d_env'] = (math.log10(10 ** -5), math.log10(10 ** 0))
limits['cell_nodes'] = (math.log10(10 ** 0), math.log10(10 ** 2))

prior = pyabc.Distribution(**{key: pyabc.RV("uniform", lb, ub - lb)
                              for key, (lb, ub) in limits.items()})

import keras
import tensorflow as tf


def generate_population_data(n_samples: int, data_dim: int, max_length: int) -> np.ndarray:
    """
    Generate population data
    :param n_samples:  number of samples
    :param data_dim:  number of cells (50) x 2 (x and y coordinates)
    :param max_length:  maximum length of the sequence
    :return:
    """
    data_raw = []
    for _ in tqdm(range(n_samples)):
        while True:
            params = prior.rvs()
            sim = model.sample(params)
            try:
                if not np.isinf(sim[0]['x']).all():
                    break
                else:
                    print("simulation was empty, try again")
            except KeyError:
                print("KeyError, since simulation was empty, try again")
        data_raw.append(sim)  # generates 50 cell in one experiment

    data = np.ones((n_samples, data_dim, max_length, 2)) * np.nan
    # each cell is of different length, each with x and y coordinates, make a tensor out of it
    for s_id, sample in enumerate(data_raw):
        for c_id, cell in enumerate(sample):
            # pre-pad the data with zeros, but first write nans to compute the mean and std
            data[s_id, c_id, -len(cell['x'][:max_length]):, 0] = cell['x'][:max_length]
            data[s_id, c_id, -len(cell['y'][:max_length]):, 1] = cell['y'][:max_length]
    return data


def normalize_data(data: np.ndarray,
                   mean: Optional[np.ndarray] = None, std: Optional[np.ndarray] = None
                   ) -> (np.ndarray, np.ndarray, np.ndarray):
    norm_data = data.copy()
    return_mean = False
    if mean is None:
        mean = np.nanmean(data[:, :, :, 0]), np.nanmean(data[:, :, :, 1])
        std = np.nanstd(data[:, :, :, 0]), np.nanstd(data[:, :, :, 1])
        return_mean = True

    norm_data[:, :, :, 0] = (data[:, :, :, 0] - mean[0]) / std[0]
    norm_data[:, :, :, 1] = (data[:, :, :, 1] - mean[1]) / std[1]
    # replace nan with -3
    if np.isnan(norm_data).any():
        print("Data has nan ..")
        print("Smallest value is ", np.nanmin(norm_data))
    norm_data[np.isnan(norm_data)] = -3
    if return_mean:
        return norm_data, mean, std
    return norm_data


sequence_length = 2 ** int(np.ceil(np.log2(1000)))  # TODO: no idea, asked emad
data_dim = 50
n_test = 2#10000
n_training = 2#100000
batch_size = 128

np.random.seed(42)
if not os.path.exists(os.path.join(gp, 'validation_data.pickle')):
    print("Generating validation data...")
    validation_data = generate_population_data(n_samples=n_test, data_dim=data_dim,
                                               max_length=sequence_length)
    # pickle the data
    with open(os.path.join(gp, 'validation_data.pickle'), 'wb') as f:
        pickle.dump(validation_data, f)
else:
    print("Loading validation data...")
    with open(os.path.join(gp, 'validation_data.pickle'), 'rb') as f:
        validation_data = pickle.load(f)

if not os.path.exists(os.path.join(gp, 'test_data.pickle')):
    print("Generating test data...")
    test_data = generate_population_data(n_samples=n_test, data_dim=data_dim,
                                         max_length=sequence_length)
    with open(os.path.join(gp, 'test_data.pickle'), 'wb') as f:
        pickle.dump(test_data, f)
else:
    print("Loading test data...")
    with open(os.path.join(gp, 'test_data.pickle'), 'rb') as f:
        test_data = pickle.load(f)

if not os.path.exists(os.path.join(gp, 'training_data.pickle')):
    print("Generating training data...")
    training_data = generate_population_data(n_samples=n_training, data_dim=data_dim,
                                             max_length=sequence_length)
    with open(os.path.join(gp, 'training_data.pickle'), 'wb') as f:
        pickle.dump(training_data, f)
else:
    print("Loading training data...")
    with open(os.path.join(gp, 'training_data.pickle'), 'rb') as f:
        training_data = pickle.load(f)

# normalize the data
validation_data, data_mean, data_std = normalize_data(validation_data)
training_data = normalize_data(training_data, mean=data_mean, std=data_std)
test_data = normalize_data(test_data, mean=data_mean, std=data_std)
# reorder the cells
np.random.shuffle(training_data)  # only shuffles first axis inplace

print("Validation data shape: ", validation_data.shape)
print("Test data shape: ", test_data.shape)
print("Training data shape: ", training_data.shape)

epochs = 100
#units_per_layer = [64, 32, 16]
dropout = 0.2



# encoder = keras.Sequential(name='encoder')
# for u_i, units in enumerate(units_per_layer):
#     if u_i == len(units_per_layer) - 1:
#         encoder.add(keras.layers.LSTM(units,
#                                       activation='tanh',
#                                       dropout=dropout,
#                                       kernel_regularizer=keras.regularizers.L2(l2=1e-3),
#                                       return_sequences=False, name=f'encoder_{u_i}'))
#     else:
#         encoder.add(keras.layers.LSTM(units,
#                                       activation='tanh',
#                                       dropout=dropout,
#                                       kernel_regularizer=keras.regularizers.L2(l2=1e-3),
#                                       return_sequences=True, name=f'encoder_{u_i}'))
#
# decoder = keras.Sequential(name='decoder')
# decoder.add(keras.layers.RepeatVector(sequence_length, name='encoder_decoder_bridge'))
# for u_i, units in enumerate(units_per_layer[::-1]):
#     decoder.add(keras.layers.LSTM(units,
#                                   activation='tanh',
#                                   dropout=dropout,
#                                   kernel_regularizer=keras.regularizers.L2(l2=1e-3),
#                                   return_sequences=True, name=f'decoder_{u_i}'))
# decoder.add(keras.layers.TimeDistributed(keras.layers.Dense(data_dim, activation='linear')))

# stack the encoder and decoder
# Create an input layer
#encoder_input = keras.layers.Input(shape=(data_dim, sequence_length, 2))

# Call the encoder model with this input
#encoder_output = encoder(encoder_input)

# Call the decoder model with the encoder output
#decoder_output = decoder(encoder_output)

# Pass the encoder output to the decoder input
#autoencoder_output = decoder(encoder_output)

# Define the autoencoder model
#autoencoder = keras.Model(inputs=encoder_input, outputs=autoencoder_output, name="autoencoder")

#input_layer = keras.layers.Input(shape=(sequence_length, data_dim))
#encoded = encoder(input_layer)
#decoded = decoder(encoded)
#autoencoder = keras.Model(inputs=input_layer, outputs=decoded, name='autoencoder')

autoencoder.compile(loss="mean_squared_error", optimizer='adam')
print(encoder.summary())
print(decoder.summary())
print(autoencoder.summary())

# early stopping
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', min_delta=1e-2, patience=5, verbose=0, mode='auto',
    baseline=None, restore_best_weights=True
)

# train the model
autoencoder.fit(x=training_data, y=training_data,
                validation_data=(validation_data, validation_data),
                epochs=epochs, batch_size=batch_size, shuffle=True, callbacks=[early_stop])

test_out = encoder.predict_on_batch(validation_data)
print(test_out.shape)

test_loss = np.mean((autoencoder.predict_on_batch(test_data) - test_data)**2)
print(f"Test loss: {test_loss}")

# save the model
autoencoder.save(gp + 'autoencoder.keras')
encoder.save(gp + 'encoder.keras')

print("Done!")
