import numpy as np
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    """A custom dataset class for custom data.

    Args:
        data (numpy.ndarray): The data array.
        data_stamp (numpy.ndarray): The data stamp array.
        data_time: The data time.
        seq_len (int): The sequence length.
        pred_len (int): The prediction length.
        gap_len (int): The gap length.
        scalers (object, optional): The scalers object. Defaults to None.

    Attributes:
        data (numpy.ndarray): The data array.
        data_stamp (numpy.ndarray): The data stamp array.
        data_time: The data time.
        seq_len (int): The sequence length.
        pred_len (int): The prediction length.
        gap_len (int): The gap length.
        scalers (object, optional): The scalers object. Defaults to None.
        return_time (bool): Whether to return the time. Defaults to False.

    """

    def __init__(self, data, data_stamp, data_time, seq_len, pred_len, gap_len, scalers=None):
        self.data = data.astype(np.float32)
        self.data_stamp = data_stamp.astype(np.float32)
        self.data_time = data_time
        self.return_time = False

        self.gap_len = gap_len
        self.pred_len = pred_len
        self.seq_len = seq_len

        self.scalers = scalers

    def __getitem__(self, index):
        """Get the item at the specified index.

        Args:
            index (int): The index of the item.

        Returns:
            list: A list containing the sequence data and prediction data: [seq_x, seq_y, seq_x_mark, seq_y_mark]

        """

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
