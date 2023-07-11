import argparse
import random

import numpy as np
import torch


def add_args():
    parser = argparse.ArgumentParser(description='TimesNet')
    # basic config
    parser.add_argument('--train', type=bool, default=True, action=argparse.BooleanOptionalAction,
                        help='train or only test the model, default True')
    parser.add_argument('--model_id', type=str, required=True, help='model id')
    parser.add_argument('--run_id', type=str, required=True, help='run id')

    parser.add_argument('--tags', type=str, default=None, help='tags separated by commas (without spaces).')
    parser.add_argument('--neptune', type=bool, default=True, action=argparse.BooleanOptionalAction,
                        help='whether to tracking training with neptune')
    parser.add_argument('--neptune_id', type=str, default=None,
                        help='neptune id, required when testing a trained model and uploading the result to neptune')
    parser.add_argument('--debug', default=False, action='store_true',
                        help="use a mini subset for 'mock' training")
    # data
    parser.add_argument('--data_path', type=str, default='./dataset/power.csv', help='data file')
    parser.add_argument('--data_source', type=str, default='B', help='data source, options:[A, B, C, AB, AC, BC, all]')
    parser.add_argument('--freq', type=str, default='t',
                        help='freq for time features encoding, '
                             'options: [s:secondly, t:minutely, h:hourly, d:daily,'
                             ' b:business days, w:weekly, m:monthly]')
    parser.add_argument('--load_min_loss', default=False, action='store_true',
                        help='load minimum loss from last training to check early stopping')
    # model input and output size
    parser.add_argument('--seq_len', type=int, default=116, help='input sequence length, default: 29h * 4samples/h')
    parser.add_argument('--gap_len', type=int, default=4, help='gap sequence length, default: 1h * 4samples/h')
    parser.add_argument('--pred_len', type=int, default=12, help='prediction sequence length, default: 3h * 4samples/h')
    # model settings
    parser.add_argument('--top_k', type=int, default=5, help='for TimesBlock')
    parser.add_argument('--num_kernels', type=int, default=6, help='for Inception Block')
    parser.add_argument('--d_model', type=int, default=64, help='dimension of model')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_ff', type=int, default=64, help='dimension of fcn')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout ratio')

    # training settings
    parser.add_argument('--iter', type=int, default=1,
                        help='perform training for the specified number of iterations with different seeds')
    parser.add_argument('--num_workers', type=int, default=2, help='number of workers for data loader')
    parser.add_argument('--max_epochs', type=int, default=50,
                        help='maximum number of training epochs if training is not early stopped')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size for data loader')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    # GPU
    parser.add_argument('--use_gpu', type=bool, action=argparse.BooleanOptionalAction, default=True, help='use gpu')
    parser.add_argument('--use_amp', type=bool, action=argparse.BooleanOptionalAction, default=True,
                        help='use automatic mixed precision training')
    return parser


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)


def set_session_id(args: argparse.Namespace, ii: int = 0):
    args.session_id = 'model_{}_run_{}@iter{}'.format(args.model_id, args.run_id, ii)
