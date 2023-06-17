import os
import time
import warnings

import neptune
import numpy as np
import pandas
import torch
import torch.nn as nn
from torch import optim

from data_provider.data_factory import custom_data_provider
from exp.expbasic import ExpBasic
from utils.metrics import metric
from utils.tools import EarlyStopping, adjust_learning_rate, plot_test

warnings.filterwarnings('ignore')


class ExpPowerForecast(ExpBasic):
    def __init__(self, args):
        super(ExpPowerForecast, self).__init__(args)
        self.writer = None
        self.provider = custom_data_provider(self.args)

    def _get_data(self, flag):
        returns = {}
        if flag not in returns:
            returns[flag] = self.provider(flag=flag)
        return returns[flag]

    def _build_model(self):
        return self.model_dict[self.args.model].Model(self.args).float()

    def _select_optimizer(self):
        return optim.Adam(self.model.parameters(), lr=self.args.learning_rate)

    def _select_criterion(self):
        return nn.MSELoss()

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, _) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)

                outputs = self.model(batch_x, batch_x_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, :, f_dim:]
                batch_y = batch_y[:, :, f_dim:]

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):

        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        self.writer = neptune.init_run(
            project="y.runyang/PowerForecast",
            api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJhMTk3Y2ZmZi05NDA1LTQ0OWEtODdhZi1lMjJiNWExYzdkMmYifQ==",
        )  # your credentials

        self.writer['args'] = self.args

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            num_batches = len(train_loader)
            for i, (batch_x, batch_y, batch_x_mark, _) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)

                outputs = self.model(batch_x, batch_x_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, :, f_dim:]
                batch_y = batch_y[:, :, f_dim:]
                loss = criterion(outputs, batch_y)
                train_loss.append(loss.item())

                self.writer['train/batch_loss'].append(loss.item())

                if (i + 1) % 100 == 0 or i == num_batches - 1:
                    print("\ti_batch: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/batch; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            self.writer['train/loss'].append(train_loss)
            self.writer['val/loss'].append(vali_loss)
            self.writer['test/loss'].append(test_loss)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test_only=False):
        test_data, test_loader = self._get_data(flag='test')

        result_df = pandas.DataFrame(index=test_data.dataset.data_time, columns=['target', 'prediction'])
        result_df.loc[:, 'target'] = test_data.dataset.data[:, -1]
        result_df.loc[:, 'prediction'] = np.nan

        if test_only:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = np.zeros((len(test_loader), self.args.pred_len))
        trues = np.zeros((len(test_loader), self.args.pred_len))
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, _, indices_x, indices_y) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)

                outputs = self.model(batch_x, batch_x_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, :, f_dim:].detach().cpu().numpy().squeeze()
                batch_y = batch_y[:, :, f_dim:].detach().cpu().numpy().squeeze()

                preds[i, :] = outputs
                trues[i, :] = batch_y

                series = result_df.iloc[indices_y[0].item():indices_y[1].item(), :]
                assert np.allclose(series.iloc[:, 0].values, batch_y.squeeze())
                # if not np.all(np.isnan(series.iloc[:, 1])):
                #     series.iloc[:, 1]=
                series.iloc[:, 1] = outputs.squeeze()

        fig = plot_test(result_df)
        if test_only:
            fig.savefig(os.path.join(folder_path, 'result.pdf'))
        else:
            self.writer['test'].upoad(fig)

        result = metric(preds, trues)
        for name, d in zip(['mae', 'mse', 'rmse', 'mape', 'mspe'], result):
            if not test_only:
                self.writer[f'test/{name}'].append(d)
            print(f'{name}: {d:.4f}')

        np.save(folder_path + 'metrics.npy', np.array([result]))
