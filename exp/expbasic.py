import os
import torch
from models import TimesNet


class ExpBasic(object):
    def __init__(self, args):
        self.args = args
        self.model_dict = {
            'TimesNet': TimesNet,
        }
        self.device = torch.device('cpu')
        self.model = self._build_model().to(self.device)

    def _build_model(self):
        raise NotImplementedError

    def _get_data(self, flag):
        pass

    def vali(self, **kwargs):
        pass

    def train(self, **kwargs):
        pass

    def test(self, **kwargs):
        pass
