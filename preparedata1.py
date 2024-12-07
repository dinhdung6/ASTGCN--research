import os
import numpy as np
import argparse
import configparser
import pandas as pd


# Function to find the indices of the data needed for prediction
def search_data(sequence_length, num_of_depend, label_start_idx,
                num_for_predict, units, points_per_hour):
    '''
    Parameters
    ----------
    sequence_length: int, total length of the data sequence
    num_of_depend: int, number of past time periods needed
    label_start_idx: int, start index of the prediction target
    num_for_predict: int, number of points to predict
    units: int, number of hours in a week/day/hour (depends on the parameter)
    points_per_hour: int, number of data points per hour

    Returns
    ----------
    list[(start_idx, end_idx)] - list of start and end indices for data extraction
    '''
    
    # Ensure points_per_hour is valid
    if points_per_hour < 0:
        raise ValueError("points_per_hour should be greater than 0!")

    # Check if the prediction index exceeds the sequence length
    if label_start_idx + num_for_predict > sequence_length:
        return None

    x_idx = []
    # Loop to find indices of the past periods (week/day/hour)
    for i in range(1, num_of_depend + 1):
        start_idx = label_start_idx - points_per_hour * units * i
        end_idx = start_idx + num_for_predict
        if start_idx >= 0:  # Ensure valid start index
            x_idx.append((start_idx, end_idx))
        else:
            return None

    if len(x_idx) != num_of_depend:  # Ensure enough dependencies are found
        return None

    return x_idx[::-1]  # Reverse the list to maintain the correct order


# Function to get samples for training/testing based on weeks, days, and hours of data
def get_sample_indices(data_sequence, num_of_weeks, num_of_days, num_of_hours,
                       label_start_idx, num_for_predict, points_per_hour=12):
    '''
    Parameters
    ----------
    data_sequence: np.ndarray, input data array (sequence_length, num_of_vertices, num_of_features)
    num_of_weeks, num_of_days, num_of_hours: int, number of weeks/days/hours to consider for history
    label_start_idx: int, start index for the prediction target
    num_for_predict: int, number of points to predict
    points_per_hour: int, data frequency per hour

    Returns
    ----------
    week_sample, day_sample, hour_sample, target: corresponding samples of past week/day/hour data and target values
    '''
    week_sample, day_sample, hour_sample = None, None, None

    # Ensure valid prediction index
    if label_start_idx + num_for_predict > data_sequence.shape[0]:
        return week_sample, day_sample, hour_sample, None

    # Get week-based sample if weeks are considered
    if num_of_weeks > 0:
        week_indices = search_data(data_sequence.shape[0], num_of_weeks,
                                   label_start_idx, num_for_predict,
                                   7 * 24, points_per_hour)  # 7 days * 24 hours = one week
        if not week_indices:
            return None, None, None, None

        week_sample = np.concatenate([data_sequence[i: j]
                                      for i, j in week_indices], axis=0)

    # Get day-based sample if days are considered
    if num_of_days > 0:
        day_indices = search_data(data_sequence.shape[0], num_of_days,
                                  label_start_idx, num_for_predict,
                                  24, points_per_hour)  # 24 hours = one day
        if not day_indices:
            return None, None, None, None

        day_sample = np.concatenate([data_sequence[i: j]
                                     for i, j in day_indices], axis=0)

    # Get hour-based sample if hours are considered
    if num_of_hours > 0:
        hour_indices = search_data(data_sequence.shape[0], num_of_hours,
                                   label_start_idx, num_for_predict,
                                   1, points_per_hour)  # 1 hour
        if not hour_indices:
            return None, None, None, None

        hour_sample = np.concatenate([data_sequence[i: j]
                                      for i, j in hour_indices], axis=0)

    # Define the target for prediction (future points)
    target = data_sequence[label_start_idx: label_start_idx + num_for_predict]

    return week_sample, day_sample, hour_sample, target


# Function to read the dataset and generate samples for model training/testing
def read_and_generate_dataset(graph_signal_matrix_filename,
                              num_of_weeks, num_of_days,
                              num_of_hours, num_for_predict,
                              points_per_hour=12, save=False):
    '''
    Parameters
    ----------
    graph_signal_matrix_filename: str, path to input dataset file
    num_of_weeks, num_of_days, num_of_hours: int, historical data to consider
    num_for_predict: int, number of points to predict
    points_per_hour: int, data frequency per hour

    Returns
    ----------
    A dictionary containing training, validation, and testing data
    '''
    data_seq = np.load(graph_signal_matrix_filename)['data']  # Load data

    all_samples = []
    # Iterate through the entire dataset and extract samples for each index
    for idx in range(data_seq.shape[0]):
        sample = get_sample_indices(data_seq, num_of_weeks, num_of_days,
                                    num_of_hours, idx, num_for_predict,
                                    points_per_hour)
        if ((sample[0] is None) and (sample[1] is None) and (sample[2] is None)):
            continue  # Skip if no valid samples are found

        week_sample, day_sample, hour_sample, target = sample
        sample = []  # Initialize the sample

        # Reshape week, day, hour samples and append to list
        if num_of_weeks > 0:
            week_sample = np.expand_dims(week_sample, axis=0).transpose((0, 2, 3, 1))  # (1, N, F, Tw)
            sample.append(week_sample)

        if num_of_days > 0:
            day_sample = np.expand_dims(day_sample, axis=0).transpose((0, 2, 3, 1))  # (1, N, F, Td)
            sample.append(day_sample)

        if num_of_hours > 0:
            hour_sample = np.expand_dims(hour_sample, axis=0).transpose((0, 2, 3, 1))  # (1, N, F, Th)
            sample.append(hour_sample)

        # Append target and timestamp
        target = np.expand_dims(target, axis=0).transpose((0, 2, 3, 1))[:, :, 0, :]  # (1, N, Tpre)
        sample.append(target)
        time_sample = np.expand_dims(np.array([idx]), axis=0)  # (1, 1)
        sample.append(time_sample)

        all_samples.append(sample)  # Append the full sample

    # Split the samples into training, validation, and testing sets
    split_line1 = int(len(all_samples) * 0.6)
    split_line2 = int(len(all_samples) * 0.8)

    # Combine samples for each set
    training_set = [np.concatenate(i, axis=0) for i in zip(*all_samples[:split_line1])]
    validation_set = [np.concatenate(i, axis=0) for i in zip(*all_samples[split_line1:split_line2])]
    testing_set = [np.concatenate(i, axis=0) for i in zip(*all_samples[split_line2:])]

    # Prepare training, validation, and testing features and targets
    train_x = np.concatenate(training_set[:-2], axis=-1)
    val_x = np.concatenate(validation_set[:-2], axis=-1)
    test_x = np.concatenate(testing_set[:-2], axis=-1)

    train_target = training_set[-2]
    val_target = validation_set[-2]
    test_target = testing_set[-2]

    train_timestamp = training_set[-1]
    val_timestamp = validation_set[-1]
    test_timestamp = testing_set[-1]

    # Normalize the data
    (stats, train_x_norm, val_x_norm, test_x_norm) = normalization(train_x, val_x, test_x)

    # Package the data into a dictionary
    all_data = {
        'train': {'x': train_x_norm, 'target': train_target, 'timestamp': train_timestamp},
        'val': {'x': val_x_norm, 'target': val_target, 'timestamp': val_timestamp},
        'test': {'x': test_x_norm, 'target': test_target, 'timestamp': test_timestamp},
        'stats': {'_mean': stats['_mean'], '_std': stats['_std']}
    }

    # Print dataset statistics
    print('train x:', all_data['train']['x'].shape)
    print('train target:', all_data['train']['target'].shape)
    print('train timestamp:', all_data['train']['timestamp'].shape)
    print()
    print('val x:', all_data['val']['x'].shape)
    print('val target:', all_data['val']['target'].shape)
    print('val timestamp:', all_data['val']['timestamp'].shape)
    print()
    print('test x:', all_data['test']['x'].shape)
    print('test target:', all_data['test']['target'].shape)
    print('test timestamp:', all_data['test']['timestamp'].shape)

    # Save the dataset if required
    if save:
        save_path = os.path.dirname(graph_signal_matrix_filename)
        filename = os.path.join(save_path, 'dataset_week%d_day%d_hour%d.npz' % (num_of_weeks, num_of_days, num_of_hours))
        np.savez_compressed(filename, train_x=all_data['train']['x'], train_target=all_data['train']['target'],
                            train_timestamp=all_data['train']['timestamp'],
                            val_x=all_data['val']['x'], val_target=all_data['val']['target'],
                            val_timestamp=all_data['val']['timestamp'],
                            test_x=all_data['test']['x'], test_target=all_data['test']['target'],
                            test_timestamp=all_data['test']['timestamp'],
                            mean=all_data['stats']['_mean'], std=all_data['stats']['_std'])

    return all_data
