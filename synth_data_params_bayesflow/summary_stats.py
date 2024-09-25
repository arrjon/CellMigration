import math
from typing import Optional

import numpy as np
import tidynamics  # to get sliding history stats in N*logN instead of N^2


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


def reduced_coordinates_to_sumstat(cell_population: np.ndarray) -> dict:
    """
    Compute the summary statistics of the reduced/visible coordinates of the cell population.

    :param cell_population: 3D array of cell populations
    :return: dictionary of summary statistics
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
    if not msd_list:
        return {'msd_mean': [np.nan], 'msd_var': [np.nan],
                'ta_mean': [np.nan], 'ta_var': [np.nan],
                'v_mean': [np.nan], 'v_var': [np.nan],
                'ad_mean': [np.nan], 'ad_var': [np.nan],
                'wt_mean': [np.nan], 'wt_var': [np.nan]}

    # get the mean of list of lists with different lengths
    msd_mean = [np.mean(x) for x in msd_list]
    ta_mean = [np.mean(x) for x in ta_list]
    v_mean = [np.mean(x) for x in v_list]
    ad_mean = [np.mean(x) for x in ad_list]
    wt_mean = [np.mean(x) for x in wt_list]

    # get the variance of list of lists with different lengths
    msd_var = [np.var(x) for x in msd_list]
    ta_var = [np.var(x) for x in ta_list]
    v_var = [np.var(x) for x in v_list]
    ad_var = [np.var(x) for x in ad_list]
    wt_var = [np.var(x) for x in wt_list]

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


def compute_mean_summary_stats(simulation_list: list[dict], remove_nan: bool = True) -> tuple:
    """
    Compute the mean summary statistics of the simulation list.

    :param
        simulation_list: list of cell populations
        remove_nan: remove nan and inf values from the averaged summary statistics of a population
    :return:
    """
    # get the valid means of the summary statistics lists
    ad_mean = []
    for i in range(len(simulation_list)):
        ad = simulation_list[i]['ad_mean']
        ad_mean.append([x for x in ad if not np.isnan(x) and not np.isinf(x)])

    msd_mean = []
    for i in range(len(simulation_list)):
        msd = simulation_list[i]['msd_mean']
        msd_mean.append([x for x in msd if not np.isnan(x) and not np.isinf(x)])

    ta_mean = []
    for i in range(len(simulation_list)):
        ta = simulation_list[i]['ta_mean']
        ta_mean.append([x for x in ta if not np.isnan(x) and not np.isinf(x)])

    vel_mean = []
    for i in range(len(simulation_list)):
        vel = simulation_list[i]['v_mean']
        vel_mean.append([x for x in vel if not np.isnan(x) and not np.isinf(x)])

    wt_mean = []
    for i in range(len(simulation_list)):
        wt = simulation_list[i]['wt_mean']
        wt_mean.append([x for x in wt if not np.isnan(x) and not np.isinf(x)])

    # compute the mean over all cells
    ad_averg = []
    for i in range(len(ad_mean)):
        ad = np.mean(ad_mean[i])
        if not remove_nan or (not np.isnan(ad) and not np.isinf(ad)):
            ad_averg.append(ad)
    ad_averg = np.array(ad_averg)

    msd_averg = []
    for i in range(len(msd_mean)):
        msd = np.mean(msd_mean[i])
        if not remove_nan or (not np.isnan(msd) and not np.isinf(msd)):
            msd_averg.append(msd)
    msd_averg = np.array(msd_averg)

    ta_averg = []
    for i in range(len(ta_mean)):
        ta = np.mean(ta_mean[i])
        if not remove_nan or (not np.isnan(ta) and not np.isinf(ta)):
            ta_averg.append(ta)
    ta_averg = np.array(ta_averg)

    vel_averg = []
    for i in range(len(vel_mean)):
        vel = np.mean(vel_mean[i])
        if not remove_nan or (not np.isnan(vel) and not np.isinf(vel)):
            vel_averg.append(vel)
    vel_averg = np.array(vel_averg)

    wt_averg = []
    for i in range(len(wt_mean)):
        wt = np.mean(wt_mean[i])
        if not remove_nan or (not np.isnan(wt) and not np.isinf(wt)):
            wt_averg.append(wt)
    wt_averg = np.array(wt_averg)

    return (ad_mean, msd_mean, ta_mean, vel_mean, wt_mean,
            ad_averg, msd_averg, ta_averg, vel_averg, wt_averg)
