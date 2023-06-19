import argparse
import random
import warnings

import numpy as np
import torch

from exp.exp_power_forecasting import ExpPowerForecast

warnings.filterwarnings('ignore')


def get_session_id(args):
    session = '{}@iter{}_seq{}_gap{}_pred{}_dm{}_el{}_df{}'.format(
        args.model_id, ii,
        args.seq_len, args.gap_len, args.pred_len,
        args.d_model,
        args.e_layers,
        args.d_ff,
    )
    args.tags = args.tags.split(',')
    if args.tags:
        session += '_(' + '_'.join(args.tags) + ')'
    return session


if __name__ == '__main__':
    fix_seed = 2021
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    parser = argparse.ArgumentParser(description='TimesNet')

    # basic config

    parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
    parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
    parser.add_argument('--model', type=str, required=False, default='TimesNet',
                        help='model name, options: [TimesNet]')
    parser.add_argument('--tags', type=str, default=None, required=False, help='tags for this training session')
    parser.add_argument('--neptune_id', type=str, default=None, required=False, help='tags for this training session')

    # data
    # parser.add_argument('--data', type=str, required=True, default='ETTm1', help='dataset name')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
    parser.add_argument('--features', type=str, default='MS',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    # parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='t',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length, default: 24h * 4samples/h')
    parser.add_argument('--gap_len', type=int, default=4, help='gap sequence length, default: 1h * 4samples/h')
    parser.add_argument('--pred_len', type=int, default=12, help='prediction sequence length, default: 3h * 4samples/h')

    # model define
    parser.add_argument('--top_k', type=int, default=5, help='for TimesBlock')
    parser.add_argument('--num_kernels', type=int, default=6, help='for Inception Block')
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
    # parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=7, help='output size')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    # parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    # parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    # parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    # parser.add_argument('--factor', type=int, default=1, help='attn factor')
    # parser.add_argument('--distil', action='store_false',
    #                     help='whether to use distilling in encoder, using this argument means not using distilling',
    #                     default=True)
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    # parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')

    # optimization
    parser.add_argument('--num_workers', type=int, default=2, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=50, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    # parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='MSE', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--use_amp', type=bool, help='use automatic mixed precision training', default=False)

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    # parser.add_argument('--gpu', type=int, default=0, help='gpu')
    # parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    # parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

    # de-stationary projector params
    parser.add_argument('--p_hidden_dims', type=int, nargs='+', default=[128, 128],
                        help='hidden layer dimensions of projector (List)')
    parser.add_argument('--p_hidden_layers', type=int, default=2, help='number of hidden layers in projector')

    args = parser.parse_args()

    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu and args.is_training else False

    args.use_amp = args.use_gpu and args.use_amp

    Exp = ExpPowerForecast

    if args.is_training:
        for ii in range(args.itr):
            # setting record of experiments
            args.session_id = get_session_id(args)

            exp = Exp(args)  # set experiments
            print('>' * 10 + f'training : {args.session_id}' + '>' * 10)
            exp.train()

            print('>' * 10 + f'testing : {args.session_id}' + '>' * 10)
            exp.test()
            torch.cuda.empty_cache()
    else:
        ii = 0
        args.session_id = get_session_id(args)

        exp = Exp(args)  # set experiments
        print('>' * 10 + f'testing : {args.session_id}' + '>' * 10)
        exp.test(test_only=True)
        torch.cuda.empty_cache()
