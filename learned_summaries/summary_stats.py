import math
from typing import Optional, Union

import numpy as np
import pandas as pd
import tidynamics  # to get sliding history stats in N*logN instead of N^2


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



def velocity(data_dict: dict) -> np.ndarray:
    """Compute the velocity of the cell."""
    x = data_dict['x']
    y = data_dict['y']
    vx = np.diff(x)
    vy = np.diff(y)
    v = np.sqrt(vx ** 2 + vy ** 2)
    return v


def MSD_tidy(data_dict: dict, x_name: str, y_name: str, all_time_lags: bool) -> Union[np.ndarray, float]:  # this should be used as it is faster
    """Compute the mean square displacement of the cell for all possible time lags.
    If nan values are present, return nan."""
    msd = tidynamics.msd(
        np.column_stack([data_dict[x_name], data_dict[y_name]]))
    if not all_time_lags:
        return msd[1]
    return msd



def MSD_nan(data_dict: dict, x_name: str, y_name: str, all_time_lags: bool) -> np.ndarray:
    """Compute the mean square displacement of the cell, handling NaN values."""
    # Extract data
    x = data_dict[x_name]
    y = data_dict[y_name]

    # Combine x and y, maintaining the original length with NaNs
    data = np.column_stack([x, y])

    # Calculate MSD using tidynamics, treating NaNs correctly
    msd = []
    for lag in range(1, len(data) if all_time_lags else 2):
        diffs = []
        for t in range(len(data) - lag):
            # Ensure both points at t and t+lag are valid
            if not (np.isnan(x[t]) or np.isnan(x[t + lag]) or np.isnan(y[t]) or np.isnan(y[t + lag])):
                dx = data[t + lag, 0] - data[t, 0]
                dy = data[t + lag, 1] - data[t, 1]
                diffs.append(dx ** 2 + dy ** 2)
        if diffs:
            msd.append(np.mean(diffs))
        else:
            msd.append(np.nan)  # If no valid pairs exist for this lag
    if len(msd) == 0:
        msd = [0]
    return np.array(msd)


def MSD(data_dict: dict, x_name="x", y_name="y", all_time_lags: bool = True) -> Union[np.ndarray, float]:
    """Compute the mean square displacement of the cell for all possible time lags."""
    if np.isnan(data_dict[x_name]).any() or np.isnan(data_dict[y_name]).any():
        return MSD_nan(data_dict, x_name, y_name, all_time_lags)
    # MSD_tidy is faster but cannot handle nans
    return MSD_tidy(data_dict, x_name, y_name, all_time_lags)


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


def compute_summary_stats(cell_population: np.ndarray) -> (list, list, list, list, list):
    """
    Compute the statistics of the reduced/visible coordinates of each cell in a cell population.

    :param cell_population: 3D array of cell populations
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
            msd_list.append(compute_mean(MSD(sim_dict, all_time_lags=False)))  # mean just for formatting
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
            msd_list.append(MSD(sim_dict, all_time_lags=True))
    msd = np.stack(msd_list, axis=0)
    msd[msd < 0] = 0  # algorithm can give smaller 0 values due to numerical impression
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
