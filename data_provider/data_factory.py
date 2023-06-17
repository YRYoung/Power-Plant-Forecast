import os

import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, random_split

from data_provider.data_sets import DatasetEttMinute, DatasetCustom
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
    data, data_stamp, data_time, scaler = csv_loader(file_path=os.path.join(args.root_path, args.data_path),
                                                     scale=True, freq=args.freq)
    full_set = DatasetCustom(
        data=data, data_stamp=data_stamp, data_time=data_time, scaler=scaler,
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
        dataset.dataset.return_time = True
        data_loader = DataLoader(
            dataset,
            batch_size=1 if istest else args.batch_size,
            shuffle=not istest,
            num_workers=args.num_workers,
            drop_last=True)

        return dataset, data_loader

    return lambda flag: provide(flag)


def csv_loader(file_path, scale, freq):
    df_raw = pd.read_csv(file_path, index_col=0)[:1000]
    df_raw.index = pd.to_datetime(df_raw.index)

    data = df_raw.values
    if scale:
        scaler = StandardScaler()
        scaler.fit(data)
        data = scaler.transform(data)
    else:
        scaler = None
    data_stamp = time_features(df_raw.index, freq=freq).transpose(1, 0)

    return data, data_stamp, df_raw.index, scaler
