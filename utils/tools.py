import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

pic_id = 0


def translate_seconds(seconds):
    minutes, secs = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02.0f}:{minutes:02.0f}:{secs:02.0f}"


def adjust_learning_rate(optimizer, epoch, args):
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))


class EarlyStopping:
    def __init__(self, save_path, patience=7, verbose=False, tolerance=0, val_loss_min=np.Inf):
        self.save_path = save_path
        self.patience = patience
        self.verbose = verbose
        self.counter = 0

        self.stop = False
        self.val_loss_min = val_loss_min
        self.tolerance = tolerance

    def __call__(self, val_loss, model, prefix):

        if np.isnan(val_loss):
            self.stop = True
        elif val_loss > self.val_loss_min + self.tolerance:
            self.counter += 1
            print(prefix + f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.stop = True
        else:

            self.save_checkpoint(val_loss, model, prefix)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, prefix):
        if self.verbose:
            print(prefix + f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model ...')
        torch.save({
            'model': model.state_dict(),
            'val_loss': val_loss,
        }, self.save_path)
        self.val_loss_min = val_loss


class EmptyWriter:
    class EmptyList:
        def __init__(self):
            pass

    def append(self, *args, **kwargs):
        pass

    def __init__(self):
        self.empty_list = self.EmptyList()

    def __getitem__(self, item):
        return self.empty_list

    def __setitem__(self, key, value):
        pass


def plot_test(result_df):
    global pic_id
    fig = plt.figure(pic_id, figsize=(20, 4))
    ax = plt.gca()
    sns.lineplot(data=result_df, palette=['red', 'blue'], ax=ax)
    pic_id += 1
    return fig


def adjustment(gt, pred):
    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred[i] = 1
    return gt, pred


def cal_accuracy(y_pred, y_true):
    return np.mean(y_pred == y_true)
