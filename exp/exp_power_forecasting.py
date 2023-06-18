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
from models import TimesNet
from utils.metrics import metric
from utils.tools import EarlyStopping, adjust_learning_rate, plot_test, translate_seconds


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

        self.writer = neptune.init_run(

            project="y.runyang/PowerForecast",
            api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJhMTk3Y2ZmZi05NDA1LTQ0OWEtODdhZi1lMjJiNWExYzdkMmYifQ==",
        )  # your credentials
        if self.args.tags:
            self.writer['sys/tags'].add(self.args.tags)
        self.writer['args'] = self.args

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            prefix = f'Epoch: {epoch} | \t'
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
                              f"batch: {i + 1}/{num_batches}, loss: {loss.item() :.5f} | {speed:.4f}s/batch")
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

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test_only=False):
        test_data, test_loader = self._get_data(flag='test')
        test_data.dataset.return_time = True

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
