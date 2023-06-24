import pickle

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, random_split

from DataProvider.datasets import DatasetCustom
from utils.timefeatures import time_features


def custom_data_provider(args):
    """
    Provide custom data for training, validation, testing, or final evaluation.

    Parameters:
    - args (object): Arguments object containing the following attributes:
        - trained (bool): Whether the model is trained or not.
        - result_path (str): Path to the result directory.
        - data_path (str): Path to the data CSV file.
        - scale (bool): Whether to perform feature scaling or not.
        - freq (str): Frequency of the time features.
        - debug (bool): Whether to enable debug mode or not.
        - seq_len (int): Length of the input sequence.
        - pred_len (int): Length of the prediction sequence.
        - gap_len (int): Length of the gap between input and prediction sequences.
        - batch_size (int): Batch size for data loading.
        - num_workers (int): Number of worker processes for data loading.

    Returns:
    - provide (function): A function that takes a flag ('train', 'val', 'test', 'final') and returns a tuple
      containing the dataset and data loader for the corresponding data split.
    """

    if args.trained:
        with open(f'{args.result_path}/scalers.pkl', 'rb') as f:
            scalers = pickle.load(f)
    else:
        scalers = None
    data, data_stamp, data_time, scalers = csv_loader(file_path=args.data_path, scale=True,
                                                      freq=args.freq, debug=args.debug, scalers=scalers)
    full_set = DatasetCustom(
        data=data, data_stamp=data_stamp, data_time=data_time, scalers=scalers,
        seq_len=args.seq_len, pred_len=args.pred_len, gap_len=args.gap_len
    )
    type_map = {'train': 0, 'val': 1, 'test': 2, 'final': 2}
    val_ratio = .2

    val_len = int(val_ratio * len(full_set))
    train_len = len(full_set) - 2 * val_len
    print(f'val/test_len: {val_len} | train_len: {train_len}')
    allsets = random_split(full_set, [train_len, val_len, val_len])

    def provide(flag):
        dataset = allsets[type_map[flag]]
        if flag == 'final':
            data_loader = DataLoader(
                dataset,
                batch_size=64,
                drop_last=True)
        else:
            data_loader = DataLoader(
                dataset,
                batch_size=args.batch_size,
                pin_memory=True,
                persistent_workers=True,
                shuffle=flag == 'train',
                num_workers=args.num_workers,
                drop_last=False)

        return dataset, data_loader

    return lambda flag: provide(flag)


def csv_loader(file_path, scale, freq, debug, scalers=None):
    """
    Load data from a CSV file and preprocess it.

    Parameters:
    - file_path (str): Path to the CSV file.
    - scale (bool): Whether to perform feature scaling or not.
    - freq (str): Frequency of the time features.
    - debug (bool): Whether to enable debug mode(using only a mini subset to train).
    - scalers (list, optional): List of pre-initialized scalers for feature scaling.

    Returns:
    - data (numpy.ndarray): Preprocessed data.
    - data_stamp (numpy.ndarray): Time features.
    - index (pandas.DatetimeIndex): Index of the data.
    - scalers (list or None): List of scalers used for feature scaling.
    """
    df_raw = pd.read_csv(file_path, index_col=0)
    df_raw.index = pd.to_datetime(df_raw.index)
    # '2020-03-01':
    df_raw = df_raw.loc[:, ['B', 'target']]
    if debug:
        df_raw = df_raw.iloc[-500:, :]

    data = df_raw.values
    if scale:
        scalers = scalers or [StandardScaler(), StandardScaler()]
        scalers[0].fit(data[:, :-1])
        scalers[1].fit(data[:, -1:])
        data = np.hstack([scalers[0].transform(data[:, :-1]), scalers[1].transform(data[:, -1:])])
    else:
        scalers = None

    data_stamp = time_features(df_raw.index, freq=freq).transpose(1, 0)

    return data, data_stamp, df_raw.index, scalers
