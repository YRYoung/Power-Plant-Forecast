import os

import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, random_split

from data_provider.data_sets import DatasetCustom
from utils.timefeatures import time_features


def custom_data_provider(args):
    data, data_stamp, data_time, scalers = csv_loader(file_path=args.data_path,
                                                      scale=True, freq=args.freq)
    full_set = DatasetCustom(
        data=data, data_stamp=data_stamp, data_time=data_time, scalers=scalers,
        seq_len=args.seq_len, pred_len=args.pred_len, gap_len=args.gap_len
    )
    type_map = {'train': 0, 'val': 1, 'test': 2}
    val_ratio = .2

    val_len = int(val_ratio * len(full_set))
    train_len = len(full_set) - 2 * val_len
    allsets = random_split(full_set, [train_len, val_len, val_len])

    def provide(flag):
        istest = flag == 'test'
        dataset = allsets[type_map[flag]]
        data_loader = DataLoader(
            dataset,
            batch_size=1 if istest else args.batch_size,
            shuffle=not istest,
            num_workers=args.num_workers,
            drop_last=True)

        return dataset, data_loader

    return lambda flag: provide(flag)


def csv_loader(file_path, scale, freq):
    df_raw = pd.read_csv(file_path, index_col=0)
    df_raw.index = pd.to_datetime(df_raw.index)
    df_raw = df_raw.loc['2020-03-01':, :]

    data = df_raw.values
    if scale:
        scalers = [StandardScaler(), StandardScaler()]
        scalers[0].fit(data[:, :3])
        scalers[1].fit(data[:, -1:])
        data = np.hstack([scalers[0].transform(data[:, :3]), scalers[1].transform(data[:, -1:])])
    else:
        scalers = None

    data_stamp = time_features(df_raw.index, freq=freq).transpose(1, 0)

    return data, data_stamp, df_raw.index, scalers
