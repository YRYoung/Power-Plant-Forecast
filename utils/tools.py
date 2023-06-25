import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

pic_id = 0


def translate_seconds(seconds):
    """
    Translate seconds into a formatted string of hours, minutes, and seconds.

    Args:
        seconds: The number of seconds to be translated.

    Returns:
        A string in the format "HH:MM:SS" representing the translated time.
    """
    minutes, secs = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02.0f}:{minutes:02.0f}:{secs:02.0f}"


def adjust_learning_rate(optimizer, epoch, args):
    """ Adjusts the learning rate of the optimizer based on the given epoch and arguments.

    Args:
        optimizer (torch.optim.Optimizer): The optimizer for which the learning rate needs to be adjusted.
        epoch (int): The current epoch.
        args (argparse.Namespace): The command line arguments.

    """
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    else:
        raise ValueError('Unknown lr adjustment type')
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))


class EarlyStopping:
    """Class for implementing early stopping during training.

    Args:
        save_path (str): The file path to save the model checkpoint.
        patience (int, optional): The number of epochs to wait for improvement before stopping. Default is 7.
        verbose (bool, optional): Whether to print messages during training. Default is False.
        tolerance (float, optional): The minimum improvement required to reset the counter. Default is 0.
        val_loss_min (float, optional): The initial minimum validation loss. Default is np.Inf.

    Attributes:
        save_path (str): The file path to save the model checkpoint.
        patience (int): The number of epochs to wait for improvement before stopping.
        verbose (bool): Whether to print messages during training.
        counter (int): The counter to keep track of the number of epochs without improvement.
        stop (bool): Whether to stop the training process.
        val_loss_min (float): The minimum validation loss achieved so far.
        tolerance (float): The minimum improvement required to reset the counter.

    """

    def __init__(self, save_path, patience, verbose=False, tolerance=0, val_loss_min=np.Inf):
        self.save_path = save_path
        self.patience = patience
        self.verbose = verbose
        self.counter = 0

        self.stop = False
        self.val_loss_min = val_loss_min
        self.tolerance = tolerance

    def __call__(self, val_loss, model, prefix):
        """Method to perform early stopping based on validation loss.

       Args:
           val_loss (float): The current validation loss.
           model (torch.nn.Module): The model being trained.
           prefix (str): The prefix string to print in the early stopping message.

       """
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
        """Method to save the model checkpoint if the validation loss improves.

        Args:
            val_loss (float): The current validation loss.
            model (torch.nn.Module): The model being trained.
            prefix (str): The prefix string to print in the save checkpoint message.

        """
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

    def stop(self):
        pass


def plot_test(result_df):
    """Plots the test results using a line plot.

    Args:
        result_df (pandas.DataFrame): The DataFrame containing the test results.

    Returns:
        matplotlib.figure.Figure: The figure object containing the line plot.

    """
    global pic_id
    fig = plt.figure(pic_id, figsize=(20, 4))
    ax = plt.gca()
    sns.lineplot(data=result_df, palette=['red', 'blue'], ax=ax)
    pic_id += 1
    return fig
