"""
*****************************************************
# -*- coding: utf-8 -*-
# @Time    : 2023/6/8 17:43
# @Author  : Chen Wang
# @Site    : 
# @File    : utils.py
# @Email   : chen.wang@ahnu.edu.cn
# @details : 
*****************************************************
"""
import numpy as np
import pandas as pd
import tensorflow as tf
import logging
import os
import datetime
from einops import repeat


# --------------------------------------------------------------------------------------------------------------- #
# dataset preprocessing
def generate_seq(data, time_embedding, num_for_predict, num_for_closeness, interval, is_multi_view):
    """
    if is_multi_view is True, the length of seq_in of x_flow comprises 3 elements:
    num_seq_in: [weekly_period, daily_trend, closeness]
    if is_multi_view is False, the length of seq_in of x_flow comprises 1 element:
    num_seq_in: [closeness]
    :param data: (num_samples, num_sensors)
    :param time_embedding: (num_samples, 2)
    :param num_for_predict:
    :param num_for_closeness:
    :param is_multi_view: Whether to process daily and weekly period pattern corresponding to prediction sequence
    :return:
    x_flow, x_time, y, y_time
    x_flow: (num_samples, num_sensors, num_seq_in)
    x_time: (num_samples, num_seq_in, 2)
    y:      (num_samples, num_sensors, num_seq_out)
    y_time: (num_samples, num_seq_out, 2)
    """
    length = data.shape[0]
    day_step = (60 // interval) * 24
    week_step = (60 // interval) * 24 * 7
    weekly_period_list = []
    daily_trend_list = []
    closeness_list = []
    x_time_list = []
    y_list = []
    y_time_list = []

    if is_multi_view:
        # 'i' denotes the first step of historical data
        for i in range(week_step - num_for_closeness, length - num_for_closeness - num_for_predict + 1):
            temp_time_list = []
            weekly_period_list.append(np.expand_dims(
                data[i - week_step + num_for_closeness: i - week_step + num_for_closeness + num_for_predict], axis=0
            ))
            temp_time_list.append(np.expand_dims(
                time_embedding[i - week_step + num_for_closeness: i - week_step + num_for_closeness + num_for_predict],
                axis=0
            ))
            daily_trend_list.append(np.expand_dims(
                data[i - day_step + num_for_closeness: i - day_step + num_for_closeness + num_for_predict], axis=0
            ))
            temp_time_list.append(np.expand_dims(
                time_embedding[i - day_step + num_for_closeness: i - day_step + num_for_closeness + num_for_predict],
                axis=0
            ))
            closeness_list.append(np.expand_dims(
                data[i: i + num_for_closeness], axis=0
            ))
            temp_time_list.append(np.expand_dims(
                time_embedding[i: i + num_for_closeness],
                axis=0
            ))      # (1, num_for_closeness, 2)
            x_time_list.append(np.concatenate(temp_time_list, axis=1))  # (1, seq_in, 2)

            y_list.append(np.expand_dims(
                data[i + num_for_closeness: i + num_for_closeness + num_for_predict], axis=0
            ))
            y_time_list.append(np.expand_dims(
                time_embedding[i + num_for_closeness: i + num_for_closeness + num_for_predict],
                axis=0
            ))      # (1, seq_out, 2)
        weekly_period = np.concatenate(weekly_period_list, axis=0)    # (num_samples, num_for_predict, num_sensors)
        daily_trend = np.concatenate(daily_trend_list, axis=0)        # (num_samples, num_for_predict, num_sensors)
        closeness = np.concatenate(closeness_list, axis=0)            # (num_samples, num_for_closeness, num_sensors)
        x_flow = np.concatenate([weekly_period, daily_trend, closeness], axis=1)
        x_flow = np.transpose(x_flow, (0, 2, 1))                      # (num_samples, num_sensors, seq_in)
        x_time = np.concatenate(x_time_list, axis=0)                  # (num_samples, seq_in, 2)
        y = np.concatenate(y_list, axis=0)                            # (num_samples, seq_out, num_sensors)
        y = np.transpose(y, (0, 2, 1))                                # (num_samples, num_sensors, seq_out)
        y_time = np.concatenate(y_time_list, axis=0)                  # (num_samples, seq_out, 2)
        return x_flow, x_time, y, y_time
    else:
        for i in range(0, length - num_for_closeness - num_for_predict + 1):
            closeness_list.append(np.expand_dims(
                data[i: i + num_for_closeness], axis=0
            ))
            x_time_list.append(np.expand_dims(
                time_embedding[i: i + num_for_closeness],
                axis=0
            ))      # (1, seq_in, 2)

            y_list.append(np.expand_dims(
                data[i + num_for_closeness: i + num_for_closeness + num_for_predict], axis=0
            ))
            y_time_list.append(np.expand_dims(
                time_embedding[i + num_for_closeness: i + num_for_closeness + num_for_predict],
                axis=0
            ))      # (1, seq_in, 2)
        closeness = np.concatenate(closeness_list, axis=0)
        x_flow = np.transpose(closeness, (0, 2, 1))                   # (num_samples, num_sensors, seq_in)
        x_time = np.concatenate(x_time_list, axis=0)                  # (num_samples, seq_in, 2)
        y = np.concatenate(y_list, axis=0)                            # (num_samples, seq_out, num_sensors)
        y = np.transpose(y, (0, 2, 1))                                # (num_samples, num_sensors, seq_out)
        y_time = np.concatenate(y_time_list, axis=0)                  # (num_samples, seq_out, 2)
        return x_flow, x_time, y, y_time


def generate_dataset(root_path, dataset, train_ratio, val_ratio, interval, num_for_predict,
                     num_for_closeness, is_multi_view=True):
    """

    :param root_path:
    :param dataset:
    :param train_ratio:
    :param val_ratio:
    :param interval: The minimized time interval of a one-time step of the dataset.
    :param num_for_predict:
    :param num_for_closeness:
    :param is_multi_view:
    :return:
    dataset_x: [train_list, val_list, test_list], each list comprises [x_flow, x_time]
    dataset_y: [train_y, val_y, test_y], the shape of x_flow, x_time and y(See `generate_seq`)
    """
    # PEMS04 shape: (16992, 307, 3)     feature: flow, occupy, speed
    # PEMS08 shape: (17856, 170, 3)     feature: flow, occupy, speed

    # load data
    data_df = pd.read_hdf(os.path.join(root_path, f'{dataset}.h5'))
    flow_data = data_df.values.astype(np.float32)

    # time embedding
    Time = data_df.index
    year = Time.year[0]
    first_day_weekday = datetime.datetime(year, 1, 1).weekday()     # 0 - 6
    # dayofyear start from 1
    weekofyear = np.reshape((Time.dayofyear - (Time.weekday - first_day_weekday) - 1) // 7 + 1, newshape=(-1, 1))
    dayofweek = np.reshape(Time.weekday, newshape=(-1, 1))
    # three digit coding,the first and second digit for weekofyear, start from 1, max is 53 ;
    # and the third digit for dayofweek, start for 0, max is 6
    dayofwork = np.where(dayofweek[..., 0: 1] <= 4, dayofweek[..., 0: 1] + 1, 0)  # value range: [0, 5]
    dayofweekend = np.where(dayofweek[..., 0: 1] > 4, dayofweek[..., 0: 1] - 4, 0)  # value range: [0, 2]
    hourofday = Time.hour.values[..., np.newaxis]
    minuteofday = Time.minute.values[..., np.newaxis]
    Time = np.concatenate([weekofyear, dayofwork, dayofweekend, hourofday, minuteofday], axis=-1)

    mean_train = None
    std_train = None
    dataset_x = []
    dataset_y = []

    all_flow_x, all_time_x, all_y, all_y_time = generate_seq(flow_data, Time, num_for_predict,
                                                             num_for_closeness, interval, is_multi_view)
    length = all_flow_x.shape[0]        # (num_samples, num_sensors, features)
    train_line, val_line = int(length * train_ratio), int(length * (train_ratio + val_ratio))

    for line1, line2 in ((0, train_line), (train_line, val_line), (val_line, length)):
        if mean_train is None:
            mean_train = all_flow_x[line1: line2].mean()
        if std_train is None:
            std_train = all_flow_x[line1: line2].std()

        x_flow, x_time = all_flow_x[line1: line2], all_time_x[line1: line2]
        y, y_time = all_y[line1: line2], all_y_time[line1: line2]
        # Z-score normalization
        x_flow = (x_flow - mean_train) / std_train

        dataset_x.append([x_flow, x_time])
        dataset_y.append([y, y_time])
    return dataset_x, dataset_y, mean_train, std_train    # need to return mean and std


# --------------------------------------------------------------------------------------------------------------- #
# metrics function
def masked_rmse_np(pred, true, null_val=np.nan):
    return np.sqrt(masked_mse_np(pred=pred, true=true, null_val=null_val))


def masked_mse_np(pred, true, null_val=np.nan):
    with np.errstate(divide='ignore', invalid='ignore'):
        if np.isnan(null_val):
            mask = ~np.isnan(true)
        else:
            mask = np.not_equal(true, null_val)
        mask = mask.astype('float32')
        mask /= np.mean(mask)
        mse = np.square(np.subtract(pred, true)).astype('float32')
        mse = np.nan_to_num(mse * mask)
        return np.mean(mse)


def masked_mape_np(pred, true, null_val=np.nan):
    with np.errstate(divide='ignore', invalid='ignore'):
        if np.isnan(null_val):
            mask = ~np.isnan(true)
        else:
            mask = np.not_equal(true, null_val)
        mask = mask.astype('float32')
        mask /= np.mean(mask)
        mape = np.abs(np.divide(np.subtract(pred, true).astype('float32'), true))
        mape = np.nan_to_num(mask * mape)
        return 100 * np.mean(mape)


def masked_mae_np(pred, true, null_val=np.nan):
    with np.errstate(divide='ignore', invalid='ignore'):
        if np.isnan(null_val):
            mask = ~np.isnan(true)
        else:
            mask = np.not_equal(true, null_val)
        mask = mask.astype('float32')
        mask /= np.mean(mask)
        mae = np.abs(np.subtract(pred, true)).astype('float32')
        mae = np.nan_to_num(mae * mask)
        return np.mean(mae)


def evaluate_all_metrics(pred, true, null_val=np.nan, logger=None):
    """

    :param pred: (n_samples, n_sensors, horizon)
    :param true: (n_samples, n_sensors, horizon)
    :param null_val: np.nan or 0.0
    :param logger:
    :return: three list(maes, rmses, mapes), len = n_horizon
    """
    horizon = true.shape[2]
    maes = []
    rmses = []
    mapes = []
    for idx in range(horizon):
        mae = masked_mae_np(pred[..., idx], true[..., idx], null_val)
        rmse = masked_rmse_np(pred[..., idx], true[..., idx], null_val)
        mape = masked_mape_np(pred[..., idx], true[..., idx], null_val)
        maes.append(mae)
        rmses.append(rmse)
        mapes.append(mape)
        if logger is None:
            print('Horizon {}: mae: {:.2f}, rmse: {:.2f}, mape: {:.2f}%.'.format(idx + 1, mae, rmse, mape))
        else:
            logger.info('Horizon {}: mae: {:.2f}, rmse: {:.2f}, mape: {:.2f}%.'.format(idx + 1, mae, rmse, mape))
    if logger is None:
        print('Average Error: mae: {:.2f}, rmse: {:.2f}, mape: {:.2f}%.'.format(np.mean(maes),
                                                                                np.mean(rmses),
                                                                                np.mean(mapes)))
    else:
        logger.info('Average Error: mae: {:.2f}, rmse: {:.2f}, mape: {:.2f}%.'.format(np.mean(maes),
                                                                                      np.mean(rmses),
                                                                                      np.mean(mapes)))
    # LaTeX code
    # print(
    #     '& {:.2f} & {:.2f} & {:.2f}\\% & {:.2f} & {:.2f} & {:.2f}\\% & {:.2f} & {:.2f} & {:.2f}\\% & {:.2f} & {:.2f} & {:.2f}\\% \\\\'.format(
    #         maes[14], rmses[14], mapes[14],  # 1 hour 15 min - 15 horizon
    #         maes[17], rmses[17], mapes[17],  # 1 hour 30 min - 18 horizon
    #         maes[23], rmses[23], mapes[23],  # 2 hour - 24 horizon
    #         np.mean(maes), np.mean(rmses), np.mean(mapes)
    #     ))

    return maes, rmses, mapes


# --------------------------------------------------------------------------------------------------------------- #
# log function
def get_logger(root, name=None, is_debug_in_screen=True, ):
    """

    :param root: the log file's root path, str
    :param name: the logger's name, str
    :param is_debug_in_screen: whether to show DEBUG in screen, bool
    :return: logging object
    """
    # when is_debug_in_screen is True, show DEBUG and INFO in screen
    # when is_debug_in_screen is False, show DEBUG in file and info in both screen & file
    # INFO will always be in screen
    # DEBUG will always be in file

    # create a logger
    logger = logging.getLogger(name)

    # critical > error > warning > info > debug > notset
    logger.setLevel(logging.DEBUG)

    # define the formate
    formatter = logging.Formatter('%(asctime)s: %(message)s', "%Y-%m-%d %H:%M")

    # create another handler for output log to console
    console_handler = logging.StreamHandler()
    if is_debug_in_screen:
        console_handler.setLevel(logging.DEBUG)
    else:
        console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    # create a handler for write log to file
    logfile = os.path.join(root, 'run.log')
    if not os.path.exists(root):
        os.makedirs(root)
    print('Creat Log File in: ', logfile)
    file_handler = logging.FileHandler(logfile, mode='w')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    # add Handler to logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger


# --------------------------------------------------------------------------------------------------------------- #
# some tools
class MemoryUsageCallback(tf.keras.callbacks.Callback):
    def __init__(self):
        super(MemoryUsageCallback, self).__init__()
        self.peak_memory = 0

    def on_epoch_end(self, epoch, logs=None):
        memory_info = tf.config.experimental.get_memory_info('GPU:0')
        current_memory = memory_info["current"]
        peak_memory = memory_info["peak"]

        print("Epoch {} - Current memory usage: {:.4f}MB, Peak memory usage: {:.4f}MB".format(epoch + 1,
                                                                                              current_memory / (
                                                                                                      1024 ** 2),
                                                                                              peak_memory / (
                                                                                                      1024 ** 2)))
        if peak_memory > self.peak_memory:
            self.peak_memory = peak_memory


class CustomCallback(tf.keras.callbacks.Callback):
    def __init__(self, logger):
        super().__init__()
        self.logger = logger
        self.start_time = None

    def on_epoch_begin(self, epoch, logs=None):
        self.start_time = datetime.datetime.now()

    def on_epoch_end(self, epoch, logs=None):
        self.logger.info("Epoch {}: Train Loss = {}, Val Loss = {}, Train Time = {} secs".format(
            epoch + 1, logs['loss'], logs['val_loss'], (datetime.datetime.now() - self.start_time).seconds
        ))


class StandardScaler:
    def __init__(self, mean, std):
        # Z-score normalization
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean
# --------------------------------------------------------------------------------------------------------------- #


if __name__ == '__main__':
    # time = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    # print(time)
    # logger = get_logger('./log', is_debug_in_screen=True)
    # logger.debug('this is a {} debug message'.format(1))
    # logger.info('this is an info message')
    # logger.debug('this is a debug message')
    # logger.info('this is an info message')
    # logger.debug('this is a debug message')
    # logger.info('this is an info message')
    dataset_x, dataset_y, mean, std = generate_dataset('./data/PEMS08', 'PEMS08', 0.6, 0.2, 5, 24, 24)
