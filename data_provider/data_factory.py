import os

import pandas as pd

from data_provider.data_loader import DatasetEttMinute, DatasetCustom
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler

from utils.timefeatures import time_features

data_dict = {
    'ETTm1': DatasetEttMinute,
    'custom': DatasetCustom,
}


def data_provider(args, flag):
    Data = data_dict[args.data]

    if flag == 'test':
        shuffle_flag = False
        drop_last = True
        batch_size = 1  # bsz=1 for evaluation
        freq = args.freq
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size  # bsz for train and valid
        freq = args.freq

    data_set = Data(
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        size=[args.seq_len, args.pred_len],
        features=args.features,
        target=args.target,
        freq=freq,
    )
    print(flag, len(data_set))
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last)
    return data_set, data_loader


def custom_data_provider(args):
    loader = csv_loader(file_path=os.path.join(args.root_path, args.data_path),
                        scale=True, freq=args.freq,
                        pred_len=args.pred_len, gap_len=args.gap_len)

    def load_in_dict(flag):
        data, data_stamp, scaler = loader(flag=flag)
        drop_last = True
        if flag == 'test':
            shuffle_flag = False
            batch_size = 1  # bsz=1 for evaluation

        else:
            shuffle_flag = True
            batch_size = args.batch_size  # bsz for train and valid

        data_set = DatasetCustom(
            data=data, data_stamp=data_stamp, scaler=scaler,
            seq_len=args.seq_len,
            pred_len=args.pred_len, gap_len=args.gap_len
        )

        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)

        return data_set, data_loader

    returns = {}

    def lambda_provide(flag):
        if flag not in returns:
            returns[flag] = load_in_dict(flag)
        return returns[flag]

    return lambda flag: lambda_provide(flag)


def csv_loader(file_path, scale, freq, pred_len, gap_len, val_ratio=.2):
    # file_path = os.path.join(self.root_path, self.data_path)
    df_raw = pd.read_csv(file_path)
    df_raw.index = pd.to_datetime(df_raw.index)

    val_len = int(val_ratio * len(df_raw))
    train_len = len(df_raw) - 2 * val_len

    ranges = [[0, train_len]]
    for i in range(1, 3):
        ranges.append([ranges[i - 1][1] - pred_len - gap_len, ranges[i - 1][1] + val_len])

    data = df_raw.values
    if scale:
        scaler = StandardScaler()
        train_data = df_raw[ranges[0][0]:ranges[0][1]]
        scaler.fit(train_data.values)
        data = scaler.transform(data)

    data_stamp = time_features(df_raw.index.values, freq=freq).transpose(1, 0)

    type_map = {'train': 0, 'val': 1, 'test': 2}

    def return_data(flag: str):
        idx = type_map[flag]
        start, end = ranges[idx]
        return data[start, end], data_stamp[start, end], (scaler if scale else None)

    return lambda flag: return_data(flag)
