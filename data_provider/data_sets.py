import numpy as np
from torch.utils.data import Dataset


class DatasetCustom(Dataset):
    def __init__(self, data, data_stamp, data_time, seq_len, pred_len, gap_len, scaler=None):
        self.data = data
        self.data_stamp = data_stamp
        self.data_time = data_time
        self.return_time = False

        self.gap_len = gap_len
        self.pred_len = pred_len
        self.seq_len = seq_len

        self.scaler = scaler

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len

        r_begin = s_end + self.gap_len
        r_end = r_begin + self.pred_len

        seq_x, seq_y = self.data[s_begin:s_end], self.data[r_begin:r_end]
        seq_x_mark, seq_y_mark = self.data_stamp[s_begin:s_end], self.data_stamp[r_begin:r_end]
        result = [seq_x, seq_y, seq_x_mark, seq_y_mark]
        if self.return_time:
            result += [[s_begin, s_end], [r_begin, r_end]]

        return result

    def __len__(self):
        return len(self.data) - self.seq_len - self.pred_len - self.gap_len + 1
