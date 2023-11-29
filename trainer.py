"""
*****************************************************
# -*- coding: utf-8 -*-
# @Time    : 2023/6/12 20:27
# @Author  : Chen Wang
# @Site    : 
# @File    : trainer.py
# @Email   : chen.wang@ahnu.edu.cn
# @details : 
*****************************************************
"""
import tensorflow as tf
import numpy as np
import pickle
import os
from utils import get_logger, evaluate_all_metrics
from utils import CustomCallback, generate_dataset, StandardScaler
from datetime import datetime
from model.tadgcn import TADGCN


class Trainer:
    def __init__(self, args):
        assert args.mode in ['train', 'test'], 'The mode should be train or test!'
        self.args = args
        self.experiment_save_path = os.path.join(self.args.save_path,
                                                 datetime.now().strftime(f'%Y-%m-%d-%H-%M-%S'))
        if not os.path.exists(self.experiment_save_path):
            os.makedirs(self.experiment_save_path)

        # init logger object, and tensorboard, logger & model save path
        self.logger = get_logger(root=self.experiment_save_path,
                                 name='{}_{}'.format(self.args.dataset,
                                                     datetime.now().strftime(f'%Y-%m-%d-%H-%M-%S')),
                                 is_debug_in_screen=self.args.debug)
        self.logger.info(f'The best model and running log will save in {self.experiment_save_path}.')

        self.tensorboard_path = os.path.join(self.experiment_save_path, 'tensorboard-logs')

        # init dataset
        self.mean, self.std, self.x_train, self.y_train, self.x_val, self.y_val, self.x_test, self.y_test, \
        self.static_node_embeddings = self.__get_dataset()
        self.scaler = StandardScaler(self.mean, self.std)

        # init mode
        if self.args.mode == 'train':
            self.is_pretrained = False
            self.model_path = None
        else:
            self.is_pretrained = True
            self.model_path = self.args.best_path

        self.model = self.__init_model()

        self.logger.info('============================================================================')
        self.logger.info('Model settings:')
        self.logger.info(str(self.args)[10: -1])
        self.logger.info('============================================================================')

    def fit(self):
        assert self.is_pretrained is False, 'The mode should be train if you want to train model!'

        def scheduler(epoch, lr):
            if epoch % self.args.decay_step == 0 and epoch != 0:
                return lr * self.args.decay_rate
            else:
                return lr

        lr = tf.keras.callbacks.LearningRateScheduler(scheduler, verbose=1)

        # memory_callback = MemoryUsageCallback()
        custom_callback = CustomCallback(self.logger)

        callbacks = [
            lr,
            # tf.keras.callbacks.TensorBoard(self.tensorboard_path),
            tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(self.experiment_save_path, 'best_model'),
                                               save_best_only=True,
                                               save_weights_only=True,
                                               monitor='val_loss',
                                               mode='min'),
            tf.keras.callbacks.EarlyStopping(patience=self.args.early_stop_patience,
                                             mode='min',
                                             monitor='val_loss'),
            custom_callback
            # memory_callback
        ]

        self.logger.info('Start Training!')
        train_start_time = datetime.now()
        # x_train: [x_flow, x_time], y_train: [y, y_time], the same as x_val, y_val, x_test, and y_test
        history = self.model.fit([self.x_train[0], self.x_train[1], self.y_train[1]], self.y_train[0],
                                 validation_data=([self.x_val[0], self.x_val[1], self.y_val[1]], self.y_val[0]),
                                 batch_size=self.args.batch_size, shuffle=self.args.shuffle,
                                 epochs=self.args.epochs, callbacks=callbacks)
        train_end_time = datetime.now()

        self.logger.info('Finish training, the total time of training is {:.2f} seconds.'.format(
            (train_end_time - train_start_time).total_seconds()
        ))

        self.logger.info('Load the best model.')
        self.__load_model_parameters(os.path.join(self.experiment_save_path, 'best_model'))
        self.logger.info('Start evaluating!')

        test_start_time = datetime.now()
        y_pred = self.model.predict([self.x_test[0], self.x_test[1], self.y_test[1]], batch_size=self.args.batch_size)
        test_end_time = datetime.now()
        self.logger.info('Finish testing, the total time of testing is {:.2f} seconds.'.format(
            (test_end_time - test_start_time).total_seconds()
        ))

        np.save(f'{self.experiment_save_path}/TADGCN_{self.args.dataset}_pred.npy', y_pred)
        np.save(f'{self.experiment_save_path}/TADGCN_{self.args.dataset}_true.npy', self.y_test[0])

        self.logger.info('============================================================================')
        total_params = sum(int(tf.size(p.numpy())) for p in self.model.trainable_weights)
        self.logger.info(f'Model total parameters: {total_params}.')
        self.logger.info(f'Finish evaluating, the performance of TADGCN on {self.args.dataset} dataset is shown below:')
        maes, rmses, mapes = evaluate_all_metrics(y_pred, self.y_test[0], 0., self.logger)
        self.logger.info('============================================================================')

        return history

    def evaluate(self):
        if self.is_pretrained is True:
            assert self.model_path is not None, 'Please give the best model path!'
            self.logger.info('Load the best model.')
            self.__load_model_parameters(self.model_path)
            self.logger.info('Start evaluating!')

            test_start_time = datetime.now()
            y_pred = self.model.predict([self.x_test[0], self.x_test[1], self.y_test[1]],
                                        batch_size=self.args.batch_size)
            test_end_time = datetime.now()
            self.logger.info('Finish testing, the total time of testing is {:.2f} seconds.'.format(
                (test_end_time - test_start_time).total_seconds()
            ))

            np.save(f'{self.experiment_save_path}/TADGCN_{self.args.dataset}_pred.npy', y_pred)
            np.save(f'{self.experiment_save_path}/TADGCN_{self.args.dataset}_true.npy', self.y_test[0])

            self.logger.info('============================================================================')
            total_params = sum(int(tf.size(p.numpy())) for p in self.model.trainable_weights)
            self.logger.info(f'Model total parameters: {total_params}.')
            self.logger.info(
                f'Finish evaluating, the performance of TADGCN on {self.args.dataset} dataset is shown below:'
            )
            maes, rmses, mapes = evaluate_all_metrics(y_pred, self.y_test[0], 0., self.logger)
            self.logger.info('============================================================================')

        elif self.is_pretrained is False:
            self.fit()
        else:
            raise ValueError('is_pretrained must be True or False!')

    def __init_model(self):
        tadgcn = TADGCN(self.static_node_embeddings, self.args)
        optimizer = tf.keras.optimizers.Adam(self.args.learning_rate)
        loss = self.mae_tf
        metrics = [self.rmse_tf]
        tadgcn.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        return tadgcn

    def __get_dataset(self):
        """
            dataset_x[0-1]: [x_flow, x_time]
            dataset_y[0-1]: [y, y_time]
        """
        # load static node embedding
        f = open(f'{self.args.root_path}/SE_{self.args.dataset}_{self.args.static_node_embed_dim}.txt', mode='r')
        lines = f.readlines()
        temp = lines[0].split(' ')
        N, dims = int(temp[0]), int(temp[1])
        static_node_embeddings = np.zeros(shape=(N, dims), dtype=np.float32)
        for line in lines[1:]:
            temp = line.split(' ')
            index = int(temp[0])
            static_node_embeddings[index] = temp[1:]

        # load dataset
        dataset_x, dataset_y, mean, std = generate_dataset(root_path=self.args.root_path,
                                                           dataset=self.args.dataset,
                                                           train_ratio=self.args.train_ratio,
                                                           val_ratio=self.args.val_ratio,
                                                           interval=self.args.time_interval,
                                                           num_for_predict=self.args.prediction_length,
                                                           num_for_closeness=self.args.closeness_length,
                                                           is_multi_view=True)
        x_train_flow, x_train_time = dataset_x[0]
        y_train, y_train_time = dataset_y[0]
        x_val_flow, x_val_time = dataset_x[1]
        y_val, y_val_time = dataset_y[1]
        x_test_flow, x_test_time = dataset_x[2]
        y_test, y_test_time = dataset_y[2]

        # process the final batch when the n_samples are not divisible exactly.(drop extra portion)
        if x_train_flow.shape[0] % self.args.batch_size > 0:
            request_train = x_train_flow.shape[0] // self.args.batch_size * self.args.batch_size
            x_train_flow = x_train_flow[: request_train, ...]
            x_train_time = x_train_time[: request_train, ...]
            y_train = y_train[: request_train, ...]
            y_train_time = y_train_time[: request_train, ...]
        if x_val_flow.shape[0] % self.args.batch_size > 0:
            request_val = x_val_flow.shape[0] // self.args.batch_size * self.args.batch_size
            x_val_flow = x_val_flow[: request_val, ...]
            x_val_time = x_val_time[: request_val, ...]
            y_val = y_val[: request_val, ...]
            y_val_time = y_val_time[: request_val, ...]
        if x_test_flow.shape[0] % self.args.batch_size > 0:
            request_test = x_test_flow.shape[0] // self.args.batch_size * self.args.batch_size
            x_test_flow = x_test_flow[: request_test, ...]
            x_test_time = x_test_time[: request_test, ...]
            y_test = y_test[: request_test, ...]
            y_test_time = y_test_time[: request_test, ...]

        self.logger.info('============================================================================')
        self.logger.info(f'The setting of dataset(name: {self.args.dataset}):')
        self.logger.info(f'mean: {mean}; std: {std}.')
        self.logger.info(f'x_train_flow shape: {x_train_flow.shape}; x_train_time shape: {x_train_time.shape}.')
        self.logger.info(f'y_train shape: {y_train.shape}; y_train_time shape: {y_train_time.shape}')
        self.logger.info(f'x_val_flow shape: {x_val_flow.shape}; x_val_time shape: {x_val_time.shape}.')
        self.logger.info(f'y_val shape: {y_val.shape}; y_val_time shape: {y_val_time.shape}')
        self.logger.info(f'x_test_flow shape: {x_test_flow.shape}; x_test_time shape: {x_test_time.shape}.')
        self.logger.info(f'y_test: {y_test.shape}; y_test_time shape: {y_test_time.shape}')
        self.logger.info(f'The number of train records is {y_train.shape[0]}.')
        self.logger.info(f'The number of val records is {y_val.shape[0]}.')
        self.logger.info(f'The number of test records is {y_test.shape[0]}.')
        self.logger.info('============================================================================')

        return mean, std, [x_train_flow, x_train_time], [y_train, y_train_time], [x_val_flow, x_val_time], \
               [y_val, y_val_time], [x_test_flow, x_test_time], [y_test, y_test_time], static_node_embeddings

    def __load_model_parameters(self, model_path):
        self.logger.info(f'Load pretrained model parameters from "{model_path}".')
        self.model.load_weights(model_path)

    def mae_tf(self, y_true, y_pred):
        return tf.keras.losses.mae(y_true, y_pred)

    def rmse_tf(self, y_true, y_pred):
        return tf.sqrt(tf.reduce_mean(tf.square(y_true - y_pred)))

    def huber_tf(self, y_true, y_pred):
        return tf.keras.losses.huber(y_true, y_pred, delta=self.args.huber_delta)


if __name__ == '__main__':
    pass
