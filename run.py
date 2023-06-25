import warnings

import torch

from Experiment.ExpPowerForecasting import ExpPowerForecast
from utils.cmd import add_args, set_seed, set_session_id

warnings.filterwarnings('ignore')

if __name__ == '__main__':

    parser = add_args()

    args = parser.parse_args()

    if args.tags:
        args.tags = args.tags.split(',')

    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu and args.train else False
    args.use_amp = args.use_gpu and args.use_amp

    print_len = 15
    if args.train:
        first_seed = 2021
        for ii in range(args.itr):
            set_seed(first_seed + ii)
            set_session_id(args, ii)

            exp = ExpPowerForecast(args)
            print('>' * print_len + f'training : {args.session_id}' + '>' * print_len)
            exp.train()

            print('>' * print_len + f'testing : {args.session_id}' + '>' * print_len)
            exp.test()
            torch.cuda.empty_cache()
    else:

        set_session_id(args)
        args.load_min_loss = True

        exp = ExpPowerForecast(args)
        print('>' * print_len + f'testing : {args.session_id}' + '>' * print_len)
        exp.test(test_only=True)
        torch.cuda.empty_cache()
