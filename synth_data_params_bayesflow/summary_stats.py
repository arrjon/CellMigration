import math
from typing import Optional

import numpy as np
import tidynamics  # to get sliding history stats in N*logN instead of N^2


# defining the summary statistics functions
def turning_angle(data_dict):
    # compute the angle between two consecutive points
    x = data_dict['x']
    y = data_dict['y']
    vx = np.diff(x)
    vy = np.diff(y)
    theta = np.arctan2(vy, vx)
    theta = np.unwrap(theta)
    theta = np.diff(theta)
    return theta


def velocity(data_dict):
    # compute the velocity of the cell
    x = data_dict['x']
    y = data_dict['y']
    vx = np.diff(x)
    vy = np.diff(y)
    v = np.sqrt(vx ** 2 + vy ** 2)
    return v


def MSD(data_dict, x_name="x", y_name="y"):
    # compute the mean square displacement of the cell
    msd = tidynamics.msd(
        np.column_stack([data_dict[x_name], data_dict[y_name]]))
    return msd


def angle_degree(data_dict):
    # compute the absolute angle between two consecutive points in degrees with respect to the x-axis
    x = data_dict['x']
    y = data_dict['y']
    vx = np.diff(x)
    vy = np.diff(y)
    list_angle_degrees = []
    for x, y in zip(vx, vy):
        list_angle_degrees.append(math.degrees(math.atan2(x, y)))
    return list_angle_degrees


def mean_waiting_time(data_dict, time_interval=30., threshold=np.pi/4):
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


def compute_mean_waiting_time(arr, time_interval=30, threshold=np.pi/4):
    num_cells, time_steps, _ = arr.shape
    mean_waiting_times = np.zeros(num_cells)

    for cell in range(num_cells):
        waiting_times = []
        last_change = 0

        # Compute initial direction
        initial_direction = arr[cell, 1] - arr[cell, 0]
        last_direction = np.arctan2(initial_direction[1], initial_direction[0])

        for t in range(2, time_steps):
            # Compute current direction
            current_vector = arr[cell, t] - arr[cell, t-1]
            current_direction = np.arctan2(current_vector[1], current_vector[0])

            # Check if direction has changed
            if abs(current_direction - last_direction) > threshold:  # 45 degree threshold
                waiting_times.append((t-1) - last_change)
                last_change = t-1

            last_direction = current_direction

        if waiting_times:
            mean_waiting_times[cell] = np.mean(waiting_times) * time_interval
        else:
            mean_waiting_times[cell] = np.nan  # If no direction change occurred

    return mean_waiting_times


# my functions
def cut_region(data_dict, x_min, x_max, y_min, y_max, return_longest) -> Optional[list[dict]]:
    """
    Cut the region of interest from the data.
    Truncate the data to the longest list if 'return_longest' is True.
    """
    x = data_dict['x']
    y = data_dict['y']
    obs_list = []
    x_cut = []
    y_cut = []
    entered = False
    for x_, y_ in zip(x, y):
        if x_min < x_ < x_max and y_min < y_ < y_max:
            # if the cell is in the region
            entered = True
            x_cut.append(x_)
            y_cut.append(y_)
        elif entered:
            # if the cell has left the region
            entered = False
            # append dicts to the lists
            obs_list.append({'x': x_cut, 'y': y_cut})
            # empty the lists
            x_cut = []
            y_cut = []
        # if the cell is outside the region, do nothing

    # append the last cell if entered
    if entered and len(x_cut) > 0:
        obs_list.append({'x': x_cut, 'y': y_cut})

    if len(obs_list) == 0:
        return None

    if return_longest:
        # get the longest list
        max_id = np.argmax([len(d['x']) for d in obs_list])
        return [obs_list[max_id]]
    return obs_list


def reduced_coordinates_to_sumstat(cell_population):
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
        return {'msd_mean': [np.inf], 'msd_var': [np.inf],
                'ta_mean': [np.inf], 'ta_var': [np.inf],
                'v_mean': [np.inf], 'v_var': [np.inf],
                'ad_mean': [np.inf], 'ad_var': [np.inf],
                'wt_mean': [np.inf], 'wt_var': [np.inf]}

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


def reduce_to_coordinates(sumstat,
                          minimal_length=0,
                          maximal_length=None,
                          only_longest_traj_per_cell=True):
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
            'y': sumstat["cell.center.y"][cell_data_idx][1:]
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
        sim_list = [{'x': sim['x'][:maximal_length], 'y': sim['y'][:maximal_length]} for sim in sim_list]
    return sim_list


def compute_mean_summary_stats(simulation_list: list[dict]) -> tuple:
    # put the 'ad' of all particles in one list
    ad_mean = []
    for i in range(len(simulation_list)):
        ad = simulation_list[i]['ad_mean']
        ad_mean.append([x for x in ad if not np.isnan(x) and not np.isinf(x)])

    MSD_mean = []
    for i in range(len(simulation_list)):
        msd = simulation_list[i]['msd_mean']
        MSD_mean.append([x for x in msd if not np.isnan(x) and not np.isinf(x)])

    TA_mean = []
    for i in range(len(simulation_list)):
        ta = simulation_list[i]['ta_mean']
        TA_mean.append([x for x in ta if not np.isnan(x) and not np.isinf(x)])

    VEL_mean = []
    for i in range(len(simulation_list)):
        vel = simulation_list[i]['v_mean']
        VEL_mean.append([x for x in vel if not np.isnan(x) and not np.isinf(x)])

    WT_mean = []
    for i in range(len(simulation_list)):
        wt = simulation_list[i]['wt_mean']
        WT_mean.append([x for x in wt if not np.isnan(x) and not np.isinf(x)])

    # get the average of ad of all particles
    ad_averg = []
    for i in range(len(ad_mean)):
        ad = np.mean(ad_mean[i])
        if not np.isnan(ad) and not np.isinf(ad):
            ad_averg.append(ad)
    MSD_averg = []
    for i in range(len(MSD_mean)):
        msd = np.mean(MSD_mean[i])
        if not np.isnan(msd) and not np.isinf(msd):
            MSD_averg.append(msd)
    TA_averg = []
    for i in range(len(TA_mean)):
        ta = np.mean(TA_mean[i])
        if not np.isnan(ta) and not np.isinf(ta):
            TA_averg.append(ta)

    VEL_averg = []
    for i in range(len(VEL_mean)):
        vel = np.mean(VEL_mean[i])
        if not np.isnan(vel) and not np.isinf(vel):
            VEL_averg.append(vel)

    WT_averg = []
    for i in range(len(WT_mean)):
        wt = np.mean(WT_mean[i])
        if not np.isnan(wt) and not np.isinf(wt):
            WT_averg.append(wt)

    return (ad_mean, MSD_mean, TA_mean, VEL_mean, WT_mean,
            ad_averg, MSD_averg, TA_averg, VEL_averg, WT_averg)
