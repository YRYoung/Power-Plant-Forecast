import argparse
import random
import warnings

import numpy as np
import torch

from Experiment.ExpPowerForecasting import ExpPowerForecast
from utils.tools import set_seed

warnings.filterwarnings('ignore')


def get_session_id(args):
    return 'model_{}_run_{}@iter{}'.format(args.model_id, args.run_id, ii)


if __name__ == '__main__':
    fix_seed = 2021
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    parser = argparse.ArgumentParser(description='TimesNet')

    # basic config
    parser.add_argument('--is_training', type=int, required=True, default=1,
                        help='train or only test the model, default True (1)')
    parser.add_argument('--model_id', type=str, required=True, help='model id')
    parser.add_argument('--run_id', type=str, required=True, help='run id')
    parser.add_argument('--tags', type=str, default=None, required=False, help='tags for this training session')
    parser.add_argument('--neptune', type=bool, default=True, required=False, action=argparse.BooleanOptionalAction,
                        help='whether to tracking training with neptune')
    parser.add_argument('--neptune_id', type=str, default=None, required=False,
                        help='neptune id, required when testing a trained model and uploading the result to neptune')
    parser.add_argument('--debug', required=False, default=False, action='store_true',
                        help='use a mini subset for sudo training')

    # data
    parser.add_argument('--data_path', type=str, default='./dataset/power.csv', help='data file')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
    parser.add_argument('--freq', type=str, default='t',
                        help='freq for time features encoding, '
                             'options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly]')
    parser.add_argument('--load_min_loss', default=False, action='store_true',
                        help='load minimum loss from last training to check early stopping')

    # for model input and output size
    parser.add_argument('--seq_len', type=int, default=116, help='input sequence length, default: 29h * 4samples/h')
    parser.add_argument('--gap_len', type=int, default=4, help='gap sequence length, default: 1h * 4samples/h')
    parser.add_argument('--pred_len', type=int, default=12, help='prediction sequence length, default: 3h * 4samples/h')

    # model define
    parser.add_argument('--enc_in', type=int, default=2, help='encoder input size')
    parser.add_argument('--c_out', type=int, default=1, help='output size')

    parser.add_argument('--top_k', type=int, default=5, help='for TimesBlock')
    parser.add_argument('--num_kernels', type=int, default=6, help='for Inception Block')

    parser.add_argument('--d_model', type=int, default=64, help='dimension of model')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_ff', type=int, default=64, help='dimension of fcn')

    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')

    # optimization
    parser.add_argument('--num_workers', type=int, default=2, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--max_epochs', type=int, default=50, help='maximum train epochs if not early stopped')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')

    # GPU
    parser.add_argument('--use_gpu', type=bool, action=argparse.BooleanOptionalAction, default=True, help='use gpu')
    parser.add_argument('--use_amp', type=bool, action=argparse.BooleanOptionalAction, default=True,
                        help='use automatic mixed precision training')

    args = parser.parse_args()

    if args.tags:
        args.tags = args.tags.split(',')

    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu and args.is_training else False
    args.use_amp = args.use_gpu and args.use_amp

    Exp = ExpPowerForecast

    print_len = 15
    if args.is_training:
        for ii in range(args.itr):
            # setting record of experiments
            args.session_id = get_session_id(args)

            exp = Exp(args)  # set experiments
            print('>' * print_len + f'training : {args.session_id}' + '>' * print_len)
            exp.train()

            print('>' * print_len + f'testing : {args.session_id}' + '>' * print_len)
            exp.test()
            torch.cuda.empty_cache()
    else:
        ii = 0
        args.session_id = get_session_id(args)

        exp = Exp(args)  # set experiments
        print('>' * print_len + f'testing : {args.session_id}' + '>' * print_len)
        exp.test(test_only=True)
        torch.cuda.empty_cache()
