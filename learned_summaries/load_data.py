import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from summary_stats import cut_region


def load_real_data(data_id: int, max_sequence_length: int, cells_in_population: int,
                   plot_data: bool = False, plot_starting_area: bool = False, plot_chemokine: bool = False):

    if not data_id in [0, 1]:
        raise ValueError("The data_id should be either 0 or 1")

    # load real data
    real_data_id = [0, 1][data_id]
    if real_data_id == 0:
        real_data_df = pd.read_csv('../real_data/37C-1_crop.csv')
        y_size = 739.79  # microns
        x_size = 279.74  # microns
        x_offset, y_offset = -7.5, -2  # correction such that the pillars are in the middle
        y_lin_shift = 0
    else:
        # more important data set, closer to what we expect
        real_data_df = pd.read_csv('../real_data/37C_ctrl2.csv')
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
            cell = cut_region(cell, x_min=316.5, x_max=856.5, y_min=1145., y_max=1351., return_longest=True)
            if cell is None:
                continue
            cell = cell[0]
        # pre-pad the data with zeros, but first write zeros as nans to compute the mean and std
        track = np.ones((max_sequence_length + 1, 2)) * np.nan
        track[-len(cell['x'][:max_sequence_length]):, 0] = cell['x'][:max_sequence_length]
        track[-len(cell['y'][:max_sequence_length]):, 1] = cell['y'][:max_sequence_length]
        real_data.append(track[1:])  # remove the first time point, same as in the simulation (often nan anyway)

    real_data = np.stack(real_data)
    real_data_full = real_data.copy()
    real_data_50 = real_data[:cells_in_population * (real_data_full.shape[0] // cells_in_population)]

    # plot the real data
    if plot_data:
        fig = plt.figure(figsize=(10, 10))
        plt.plot(real_data_full[0, :, 0], real_data_full[0, :, 1], 'r', label='Real Cell Trajectories')
        for cell_id in range(1, real_data_full.shape[0]):
            plt.plot(real_data_full[cell_id, :, 0], real_data_full[cell_id, :, 1], 'r')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.plot([415 / 1.31, 415 / 1.31], [1145, 1351], 'black',
                 label='visible window (in simulations)')  # left vertical line
        plt.plot([856.5, 856.5], [1145, 1351], 'black')  # right vertical line
        plt.plot([415 / 1.31, 856.5], [1145, 1145], 'black')  # lower horizontal line
        plt.plot([415 / 1.31, 856.5], [1351, 1351], 'black')  # upper horizontal line

        # plot starting area
        if plot_starting_area:
            #starting_center = 1173./2, 1500./ factor / 2
            starting_center = 1173. / 2, 2500. / 3
            starting_radius = 65 / factor
            circle = plt.Circle(starting_center, starting_radius, color='lightblue', fill=False, label='Random Starting Area')
            ax = plt.gca()
            ax.add_patch(circle)
        #
        # # add chemokine
        if plot_chemokine:
            starting_center = 1173./2, (1500+1500/2+270)/ factor
            starting_radius = 550
            circle = plt.Circle(starting_center, starting_radius, color='orange', alpha=0.5, fill=True, label='Chemokine (std)')
            ax = plt.gca()
            ax.add_patch(circle)

        plt.plot([window_x1, window_x1], [window_y1, window_y2], color='grey', label='actual window')  # left vertical line
        plt.plot([window_x2, window_x2], [window_y1, window_y2], color='grey')  # right vertical line
        plt.plot([window_x1, window_x2], [window_y1, window_y1], color='grey')  # lower horizontal line
        plt.plot([window_x1, window_x2], [window_y2, window_y2], color='grey')  # upper horizontal line
        tiff_im = plt.imread('Cell_migration_grid_v3_final2_invers.tiff')
        plt.imshow(tiff_im, origin='lower')
        plt.legend()
        plt.show()
    return real_data_50, real_data_full
