Basic Configuration

- `--train`: Specify whether to train or only test the model. Default: True (train)
- `--model_id`: Specify the model ID.
- `--run_id`: Specify the run ID.
- `--tags`: Specify tags for this training session. Each tag should be separated by a comma (without spaces). (optional)
- `--neptune`: Specify whether to track training with Neptune. Default: True
- `--neptune_id`: Specify the Neptune ID, required when testing a trained model and uploading the result to Neptune. (
  optional)
- `--debug`: Use a mini subset for 'mock' training. (optional)

Data

- `--data_path`: Specify the path to the data file. Default: ./dataset/power.csv
- `--data_source`: Specify the data source. Options: [A, B, C, AB, AC, BC, all]. Default: 'B'
- `--checkpoints`: Specify the location of model checkpoints. Default: ./checkpoints/
- `--freq`: Specify the frequency for time features encoding.
  Options: [s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly]. Default: t
- `--load_min_loss`: Load minimum loss from last training to check early stopping. set True for early-stop function when
  resuming training on the same data. Default: False

Model Input and Output Size

- `--seq_len`: Specify the input sequence length. Default: 116 (29h * 4samples/h)
- `--gap_len`: Specify the gap sequence length. Default: 4 (1 * 4samples/h)
- `--pred_len`: Specify the prediction sequence length. Default: 12 (3 * 4samples/h)

Model Settings

- `--top_k`: Specify the parameter for TimesBlock. Default: 5
- `--num_kernels`: Specify the parameter for Inception Block. Default: 6
- `--d_model`: Specify the dimension of the model. Default: 64
- `--e_layers`: Specify the number of encoder layers. Default: 2
- `--d_ff`: Specify the dimension of the fully connected network. Default: 64
- `--dropout`: Specify the dropout rate. Default: 0.1

Training Settings

- `--itr`: Perform training for the specified number of iterations with different seeds. Default: 1
- `--num_workers`: Specify the number of workers for data loader. Default: 2
- `--max_epochs`: Specify the maximum number of training epochs if not early stopped. Default: 50
- `--batch_size`: Specify the batch size of data loader. Default: 64
- `--patience`: Specify the number of times the algorithm will tolerate not seeing any improvement in validation loss
  before early-stopping. Default: 3
- `--learning_rate`: Specify the optimizer learning rate. Default: 0.0001

GPU

- `--use_gpu`: Specify whether to use GPU. Default: True
- `--use_amp`: Specify whether to use automatic mixed precision training, if set to True, `seq_len + pred_len` must be a
  power of 2. Default: True

