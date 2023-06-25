import json
import os
import pickle
import time

import neptune
import numpy as np
import pandas
import pandas as pd
import torch
import torch.nn as nn
import yaml
from torch import optim
from tqdm import tqdm

from DataProvider.data_factory import custom_data_provider
from models import TimesNet
from utils.cmd import set_session_id, add_args
from utils.metrics import metric
from utils.timefeatures import time_features
from utils.tools import EarlyStopping, plot_test, translate_seconds, EmptyWriter


class ExpPowerForecast:
    """Experiment class for power forecasting.
    Args:
        args (argparse.Namespace): Arguments dictionary.

    Attributes:
        args (argparse.Namespace): Arguments dictionary.
        writer (neptune.experiments.Experiment): Neptune data writer.
        device (torch.device): Device to use for training.
        model (torch.nn.Module): Model to train/test.
        criterion (torch.nn.Module): Loss function.
        best_model_path (str): Path to the best model checkpoint.
        early_stopping (utils.tools.EarlyStopping): Early stopping object.
        provider (function): Data provider function.

    """

    def __init__(self, args):

        self.args = args
        self.writer = None
        self.scalers = None

        device_name = 'cuda' if args.use_gpu else 'cpu'
        self.device = torch.device(device_name)

        self.args.enc_in = len(list(args.data_source)) + 1
        self.args.c_out = 1

        self.model = TimesNet.Model(self.args).float()

        self.best_model_path = self._set_path()

        self.early_stopping = self._set_early_stopping(device_name)

        self.model.to(self.device)

        self.provider = custom_data_provider(self.args)

        self.criterion = nn.MSELoss()

    def _set_early_stopping(self, device_name):
        val_loss = np.Inf
        args = self.args
        print(f'{"Trained" if args.trained else "New"} model loaded on {device_name}')
        if args.trained:
            previous_checkpoint = f'checkpoint_{args.trained - 1}.pth'
            checkpoint = torch.load(self.args.checkpoint_dir + previous_checkpoint, map_location=self.device)
            self.model.load_state_dict(checkpoint['model'])
            info = f'Previous checkpoint: {previous_checkpoint}'
            if args.load_min_loss:
                val_loss = checkpoint['val_loss']
                info += f' | minimum val_loss: {val_loss:.6f}'
            print(info)
        return EarlyStopping(save_path=self.best_model_path, val_loss_min=val_loss,
                             patience=args.patience, verbose=True)

    def _set_path(self):
        self.args.result_path = f'./results/{self.args.session_id}/'
        self.args.checkpoint_dir = self.args.result_path + 'checkpoints/'
        self.args.test_path = self.args.result_path + 'test/'
        os.makedirs(self.args.checkpoint_dir, exist_ok=True)
        os.makedirs(self.args.test_path, exist_ok=True)
        self.args.trained = len(os.listdir(self.args.checkpoint_dir))
        return os.path.join(self.args.checkpoint_dir, f'checkpoint_{self.args.trained}.pth')

    def _get_data(self, flag):
        returns = {}
        if flag not in returns:
            returns[flag] = self.provider(flag=flag)
        return returns[flag]

    def _select_optimizer(self):
        return optim.Adam(self.model.parameters(), lr=self.args.learning_rate)

    def _init_writer(self):
        with open("neptune.yaml", 'r') as stream:
            config = yaml.safe_load(stream)

        self.writer = neptune.init_run(
            with_id=self.args.neptune_id,
            project=config['project'],
            api_token=config['api_token']
        ) if self.args.neptune else EmptyWriter()

    def validation(self, data_loader) -> float:
        """Perform validation on the model.

        Args:
            data_loader (DataLoader): Validation data loader.

        Returns:
            float: The validation loss.
        """
        total_loss = np.zeros(len(data_loader))

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(data_loader):
                if self.args.use_gpu:
                    batch_x = batch_x.cuda(non_blocking=True)
                    batch_y = batch_y.cuda(non_blocking=True)
                    batch_x_mark = batch_x_mark.cuda(non_blocking=True)
                    batch_y_mark = batch_y_mark.cuda(non_blocking=True)

                with torch.cuda.amp.autocast(self.args.use_amp):
                    outputs = self.model(batch_x, batch_x_mark, batch_y[..., :-1], batch_y_mark)

                    outputs = outputs[:, :, -1:]
                    batch_y = batch_y[:, :, -1:]

                    loss = self.criterion(outputs, batch_y)

                total_loss[i] = loss.item()
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self):
        """Train the model.

        Returns:
            The trained model.
        """

        train_data, train_loader = self._get_data(flag='train')
        val_data, val_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        model_optim = self._select_optimizer()

        self._init_writer()
        if self.args.tags:
            self.writer['sys/tags'].add(self.args.tags)
        self.writer['args'] = self.args

        time_now = time.time()
        for epoch in range(self.args.max_epochs):
            iter_count = 0
            prefix = f'Epoch: {epoch} | '
            num_batches = len(train_loader)
            train_loss = np.zeros(num_batches)

            self.model.train()
            epoch_time = time.time()

            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                if self.args.use_gpu:
                    batch_x = batch_x.cuda(non_blocking=True)
                    batch_y = batch_y.cuda(non_blocking=True)
                    batch_x_mark = batch_x_mark.cuda(non_blocking=True)
                    batch_y_mark = batch_y_mark.cuda(non_blocking=True)

                with torch.cuda.amp.autocast(self.args.use_amp):

                    outputs = self.model(batch_x, batch_x_mark, batch_y[..., :-1], batch_y_mark)
                    outputs = outputs[:, :, -1:]
                    batch_y = batch_y[:, :, -1:]
                    loss = self.criterion(outputs, batch_y)

                train_loss[i] = loss.item()

                if (i + 1) % 10 == 0:
                    self.writer['train/batch_loss'].append(loss.item())

                    if (i + 1) % 100 == 0:
                        speed = (time.time() - time_now) / iter_count
                        print(prefix +
                              f"\tbatch: {i + 1}/{num_batches}, loss: {loss.item() :.5f} | {speed:.4f}s/batch")
                        iter_count = 0
                        time_now = time.time()

                loss.backward()
                model_optim.step()

            train_loss = np.average(train_loss)

            print(prefix + 'validating...', end='')
            t = time.time()
            vali_loss = self.validation(val_loader)
            test_loss = self.validation(test_loader)
            print(f' | {translate_seconds(time.time() - t)}')

            self.writer['train/loss'].append(train_loss)
            self.writer['val/loss'].append(vali_loss)
            self.writer['test/loss'].append(test_loss)

            print(prefix + "Train Loss: {0:.7f}, Vali Loss: {1:.7f}, Test Loss: {2:.7f} | {3}".format(
                train_loss, vali_loss, test_loss,
                translate_seconds(time.time() - epoch_time)))
            self.early_stopping(vali_loss, self.model, prefix)
            if self.early_stopping.stop:
                print(prefix + "Early stopping")
                break

        checkpoint = torch.load(self.best_model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model'])

        return self.model

    def test(self, test_only=False):
        """Test the model.

        Testing starts by initializing the writer and setting the device to CPU,
        and then obtaining the test data and loader。
        The model is set to evaluation mode and predictions are made batch-wise and stored in `result_df`.
        The target and prediction values are de-standardized and plotted.
        Evaluation metrics are then calculated and stored in the writer and the `metric.json`.
        The `result_df` is stored in `result.csv`.

        Args:
            test_only: Whether to only test the model without training.

        Raises:
            ValueError: If neptune id is not specified.

        Returns:
            The trained model.
        """

        if test_only:
            if self.args.neptune_id is None and self.args.neptune:
                raise ValueError('Specify neptune id')
            self._init_writer()

        self.device = torch.device('cpu')
        self.model = self.model.to(self.device)
        print('Testing on cpu')

        test_data, test_loader = self._get_data(flag='final')
        test_data.dataset.return_time = True

        result_df = pandas.DataFrame(index=test_data.dataset.data_time, columns=['target', 'prediction'])
        result_df.loc[:, 'target'] = test_data.dataset.data[:, -1]
        result_df.loc[:, 'prediction'] = np.nan

        num_batches = len(test_loader)
        prediction = np.zeros((num_batches, test_loader.batch_size, self.args.pred_len))
        target = np.zeros((num_batches, test_loader.batch_size, self.args.pred_len))

        self.model.eval()
        pred_time = 0
        with torch.no_grad():
            for i, (batch_x, batch_y,
                    batch_x_mark, batch_y_mark,
                    indices_x, indices_y) in tqdm(enumerate(test_loader), total=num_batches, desc='Test', unit='batch'):
                t_start = time.time()
                outputs = self.model(batch_x, batch_x_mark, batch_y[..., :-1], batch_y_mark)
                pred_time += time.time() - t_start
                outputs = outputs[:, :, -1:].numpy().squeeze()
                batch_y = batch_y[:, :, -1:].numpy().squeeze()

                prediction[i, :] = outputs
                target[i, :] = batch_y

                for j in range(test_loader.batch_size):
                    result_df.iloc[indices_y[0][j].item():indices_y[1][j].item(), 1] = outputs[j, :]

        pred_time /= num_batches
        print(f'Average Interference speed: {pred_time:.5f}sample/s\n')

        self.writer['pred_time'] = pred_time
        result_df_original = result_df.copy(deep=True)

        de_standardize = test_data.dataset.scalers[1].inverse_transform
        result_df_original.iloc[:, 0] = de_standardize(result_df_original.iloc[:, [0]])
        result_df_original.iloc[:, 1] = de_standardize(result_df_original.iloc[:, [1]])

        fig = plot_test(result_df_original)

        self.writer['result'].append(fig)
        fig.savefig(os.path.join(self.args.test_path, f'result.pdf'))

        prediction = de_standardize(prediction.reshape(-1, 1))
        target = de_standardize(target.reshape(-1, 1))

        result = dict()
        result['de_standardized'] = metric(prediction, target)
        result['clipped'] = metric(np.clip(prediction, target.min(), target.max()), target)
        result['30min_averaged'] = metric(prediction.reshape(-1, 2).mean(axis=1), target.reshape(-1, 2).mean(axis=1))

        for k, dic in result.items():
            print(k + ':')
            for key, value in dic.items():
                self.writer[f'{k}/{key}'] = value
                print(f'{key}: {value:.4f}')

        with open(f'{self.args.test_path}/metrics.json', 'w', encoding='utf8') as json_file:
            json.dump(result, json_file, ensure_ascii=False)

        with open(f'{self.args.result_path}/scalers.pkl', 'wb') as f:
            pickle.dump(test_data.dataset.scalers, f)

        result_df_original.to_csv(os.path.join(self.args.test_path, 'result_df.csv'))

        self.writer.stop()
