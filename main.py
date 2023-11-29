"""
*****************************************************
# -*- coding: utf-8 -*-
# @Time    : 2023/6/8 20:09
# @Author  : Chen Wang
# @Site    : 
# @File    : main.py
# @Email   : chen.wang@ahnu.edu.cn
# @details : current: Version 24
*****************************************************
"""
import numpy as np
import tensorflow as tf
import argparse
import yaml
from trainer import Trainer


if __name__ == '__main__':
    # get input args and reset
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, choices=['PEMS04', 'PEMS08'],
                        default='PEMS08', help='')
    parser.add_argument('--closeness_use_dynamic', type=str, choices=['T', 'F'],
                        default=None, help='')
    parser.add_argument('--trend_use_dynamic', type=str, choices=['T', 'F'],
                        default=None, help='')
    parser.add_argument('--period_use_dynamic', type=str, choices=['T', 'F'],
                        default=None, help='')

    args_input = parser.parse_args()

    # run settings
    DATASET = args_input.dataset      # PEMS04, PEMS08

    # load config
    config_file = './configs/{}.yaml'.format(str.upper(DATASET))
    config = yaml.load(
        open(config_file),
        Loader=yaml.FullLoader
    )

    # parser
    args = argparse.Namespace(**config)

    # init gpu config
    if args.cuda:
        gpus = tf.config.list_physical_devices(device_type='GPU')
        tf.config.experimental.set_visible_devices(devices=gpus[args.device], device_type='GPU')
        if args.is_limit:
            tf.config.experimental.set_virtual_device_configuration(
                gpus[0],
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=args.max_memory_usage)]
            )  # set the max gpu memory to allocate
        elif args.is_dynamic_allocation:
            tf.config.experimental.set_memory_growth(gpus[0], True)
    else:
        cpus = tf.config.list_physical_devices(device_type='CPU')
        tf.config.experimental.set_visible_devices(devices=cpus[0], device_type='CPU')

    if args_input.closeness_use_dynamic is not None:
        args.closeness_use_dynamic = True if args_input.closeness_use_dynamic == 'T' else False

    if args_input.trend_use_dynamic is not None:
        args.trend_use_dynamic = True if args_input.trend_use_dynamic == 'T' else False

    if args_input.period_use_dynamic is not None:
        args.period_use_dynamic = True if args_input.period_use_dynamic == 'T' else False

    # init trainer
    trainer = Trainer(args)

    if args.mode == 'train':
        # train
        history = trainer.fit()
    elif args.mode == 'test':
        # test
        trainer.evaluate()
    else:
        raise ValueError()
