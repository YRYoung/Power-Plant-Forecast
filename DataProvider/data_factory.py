import pickle

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Subset

from DataProvider.datasets import CustomDataset
from utils.timefeatures import time_features


def custom_data_provider(args):
    """A custom data provider function.

    Args:
        args (argparse.Namespace): The arguments dict.

    Returns:
        function: A lambda function that provides the dataset and data loader, with a given flag.

    Raises:
        FileNotFoundError: If the file path specified in args.data_path does not exist.

"""

    if args.trained:
        with open(f'{args.result_path}/scalers.pkl', 'rb') as f:
            scalers = pickle.load(f)
    else:
        scalers = None
    data, data_stamp, data_time, scalers = csv_loader(file_path=args.data_path, scale=True, source=args.data_source,
                                                      freq=args.freq, debug=args.debug, scalers=scalers)
    full_set = CustomDataset(
        data=data, data_stamp=data_stamp, data_time=data_time, scalers=scalers,
        seq_len=args.seq_len, pred_len=args.pred_len, gap_len=args.gap_len
    )
    type_map = {'train': 0, 'val': 1, 'test': 2, 'final': 2}
    val_ratio = .2

    val_len = int(val_ratio * len(full_set))
    train_len = len(full_set) - 2 * val_len
    print(f'val/test_len: {val_len} | train_len: {train_len}')

    indices = np.arange(len(full_set))
    all_sets = [Subset(full_set, i) for i in [indices[:train_len],
                                              indices[train_len:train_len + val_len],
                                              indices[train_len + val_len:]]]

    def provide(flag):
        """Provide the dataset and data loader based on the flag.

        Args:
            flag (str): The flag indicating the type of dataset.

        Returns:
            tuple: A tuple containing the dataset and data loader.

        """
        dataset = all_sets[type_map[flag]]
        if flag == 'final':
            data_loader = DataLoader(
                dataset,
                batch_size=args.batch_size,
                shuffle=False,
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


def csv_loader(file_path, scale, freq, debug, scalers=None, source='B'):
    """ Load data from a CSV file and preprocess it for further analysis.

    Parameters:
        file_path (str): The path to the CSV file.
        scale (bool): Whether to scale the data or not.
        freq (str): The frequency of the data timestamps.
        debug (bool): Whether to run in debug mode or not.
        scalers (list, optional): List of scalers to use for scaling the data. Defaults to None.
        source (str): The source of the data. Defaults to 'B'.

    Returns:
        tuple: A tuple containing the preprocessed data, the timestamp data,
        the index (date) of the original data, and the scalers used for scaling.

    """

    df_raw = pd.read_csv(file_path, index_col=0)
    df_raw.index = pd.to_datetime(df_raw.index)
    # '2020-03-01':
    if source.lower() != 'all':
        columns = list(source) + ['target']
        print(f'Using data {columns}')
        df_raw = df_raw.loc[:, columns]
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
