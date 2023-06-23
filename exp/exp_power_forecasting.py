import os
import time

import neptune
import numpy as np
import pandas
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm

from data_provider.data_factory import custom_data_provider
from models import TimesNet
from utils.metrics import metric
from utils.tools import EarlyStopping, plot_test, translate_seconds


class ExpPowerForecast():
    def __init__(self, args):
        self.args = args
        self.writer = None
        self.provider = custom_data_provider(self.args)

        device_name = 'cuda' if args.use_gpu else 'cpu'
        self.device = torch.device(device_name)
        self.model = TimesNet.Model(self.args).float()

        path = os.path.join(self.args.checkpoints, self.args.session_id)

        os.makedirs(path, exist_ok=True)

        self.best_model_path = os.path.join(path, 'checkpoint.pth')
        trained = os.path.exists(self.best_model_path)
        val_loss = np.Inf
        print(f'{"Trained" if trained else "New"} model loaded on {device_name}')
        if trained:
            checkpoint = torch.load(self.best_model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model'])
            val_loss = checkpoint['val_loss']
            print(f'Previous minimum val_loss: {val_loss:.6f}')

        self.early_stopping = EarlyStopping(save_path=self.best_model_path, val_loss_min=val_loss,
                                            patience=self.args.patience, verbose=True)

        self.model.to(self.device)

    def _get_data(self, flag):
        returns = {}
        if flag not in returns:
            returns[flag] = self.provider(flag=flag)
        return returns[flag]

    def _select_optimizer(self):
        return optim.Adam(self.model.parameters(), lr=self.args.learning_rate)

    def _select_criterion(self):
        return nn.MSELoss()

    def vali(self, vali_loader, criterion):
        total_loss = np.zeros(len(vali_loader))

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, _) in enumerate(vali_loader):
                if self.args.use_gpu:
                    batch_x = batch_x.cuda(non_blocking=True)
                    batch_y = batch_y.cuda(non_blocking=True)
                    batch_x_mark = batch_x_mark.cuda(non_blocking=True)

                with torch.cuda.amp.autocast(self.args.use_amp):
                    outputs = self.model(batch_x, batch_x_mark)

                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, :, f_dim:]
                    batch_y = batch_y[:, :, f_dim:]

                    loss = criterion(outputs, batch_y)

                total_loss[i] = loss.item()
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self):

        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        time_now = time.time()

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        self._init_writer()
        if self.args.tags:
            self.writer['sys/tags'].add(self.args.tags)
        self.writer['args'] = self.args

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            prefix = f'Epoch: {epoch} | '
            num_batches = len(train_loader)
            train_loss = np.zeros(num_batches)

            self.model.train()
            epoch_time = time.time()

            for i, (batch_x, batch_y, batch_x_mark, _) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                if self.args.use_gpu:
                    batch_x = batch_x.cuda(non_blocking=True)
                    batch_y = batch_y.cuda(non_blocking=True)
                    batch_x_mark = batch_x_mark.cuda(non_blocking=True)

                with torch.cuda.amp.autocast(self.args.use_amp):

                    outputs = self.model(batch_x, batch_x_mark)
                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, :, f_dim:]
                    batch_y = batch_y[:, :, f_dim:]
                    loss = criterion(outputs, batch_y)

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
            vali_loss = self.vali(vali_loader, criterion)
            test_loss = self.vali(test_loader, criterion)
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

    def _init_writer(self):
        with open("neptune_config.yaml", 'r') as stream:
            config = yaml.safe_load(stream)

        self.writer = neptune.init_run(
            with_id=self.args.neptune_id,
            project=config['project'],
            api_token=config['api_token']
        ) if self.args.neptune else EmptyWriter()

    def test(self, test_only=False):

        if test_only:
            if self.args.neptune_id is None:
                raise ValueError('Specify neptune id')
            self._init_writer()

        self.device = torch.device('cpu')
        self.model = self.model.to(self.device)

        test_data, test_loader = self._get_data(flag='final')
        test_data.dataset.return_time = True

        result_df = pandas.DataFrame(index=test_data.dataset.data_time, columns=['target', 'prediction'])
        result_df.loc[:, 'target'] = test_data.dataset.data[:, -1]
        result_df.loc[:, 'prediction'] = np.nan

        num_batches = len(test_loader)
        preds = np.zeros((num_batches, test_loader.batch_size, self.args.pred_len))
        trues = np.zeros((num_batches, test_loader.batch_size, self.args.pred_len))

        result_path = './results/' + self.args.session_id + '/'
        if not os.path.exists(result_path):
            os.makedirs(result_path)

        self.model.eval()
        pred_time = 0
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, _, indices_x, indices_y) in tqdm(enumerate(test_loader),
                                                                                     total=num_batches, desc='Test'):
                t_start = time.time()
                outputs = self.model(batch_x, batch_x_mark)
                pred_time += time.time() - t_start
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, :, f_dim:].numpy().squeeze()
                batch_y = batch_y[:, :, f_dim:].numpy().squeeze()

                preds[i, :] = outputs
                trues[i, :] = batch_y

                for i in range(test_loader.batch_size):
                    result_df.iloc[indices_y[0][i].item():indices_y[1][i].item(), 1] = outputs[i, :]

        pred_time /= num_batches
        print(f'Average Interference speed: {pred_time:.5f}sample/s')
        if not test_only:
            self.writer['pred_time'] = pred_time
        result_df_original = result_df.copy(deep=True)

        result_df_original.iloc[:, 0] = test_data.dataset.scalers[1].inverse_transform(result_df_original.iloc[:, [0]])
        result_df_original.iloc[:, 1] = test_data.dataset.scalers[1].inverse_transform(result_df_original.iloc[:, [1]])

        figs = plot_test(result_df), plot_test(result_df_original)
        for i, fig in enumerate(figs):
            if not test_only:
                self.writer['result'].append(fig)
            fig.savefig(os.path.join(result_path, f'result_{i}.pdf'))

        result = metric(preds, trues)
        for name, d in zip(['mae', 'mse', 'rmse', 'mape', 'mspe'], result):
            if not test_only:
                self.writer[f'test/{name}'].append(d)
            print(f'{name}: {d:.4f}')

        np.save(result_path + 'metrics.npy', np.array([result]))
        result_df_original.to_csv(os.path.join(result_path, f'result_df.csv'))

        if not test_only:
            self.writer.stop()
