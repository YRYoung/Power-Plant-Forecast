# TimesNet for Forecasting Power Plant Output

<!-- start intro -->
This project is an improved version of the state-of-the-art [TimesNet](https://arxiv.org/abs/2210.02186) model for time
series analysis.

It is a temporal 2D-variation modeling approach for general time series analysis:

- It extends the analysis of temporal variations into the 2D space by transforming the 1D time series into a matrix,
  which encodes temporal features from multiple perspectives.
- It uses a pyramid-like architecture, TimesBlock, to hierarchically capture the complex temporal variations.
- The experiments on a variety of tasks demonstrate that TimesNet outperforms existing methods in terms of accuracy and
  generalization capability on datasets with diverse scales and complexities.

Original Repository is [here](https://github.com/thuml/Time-Series-Library)

### 🛠️ Improvements

- Optimized and simplified the project structure from the original library.
- Implemented `Neptune` for experiment tracking.
- Enhanced dataset and data loader flexibility to handle tasks with time gaps
- Implemented parallel loading and improved data structures to enhance data processing efficiency.
- Identified and resolved a minor bug in dataset length calculation, resulting in an increased dataset size.
- Integrated additional sub-networks to process time and temperature information for prediction (e.i., `y_data`).
- Accelerated the training process by utilizing `PyTorch` built-in automatic mixed precision training and asynchronous
  GPU data copying when a GPU is available.

### 🌟 Features

- Mean Squared Error (MSE) of minute-level prediction <= **0.11** (standardized data)
- Precise and adjustable **minute-level** predictions
- Fast Inference: approximately **5s/32samples** (each sample contains 4 power output within 1 hour) on Google Colab CPU

<!-- end intro -->

## Installation

<!-- start installation -->

1. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```
2. (Optional) Config the `neptune.yaml` file for Neptune.ai tracking:
   ```
   project: your_username/your_project_name
   api_token: your_api_token
   ```

<!-- end installation -->

<!-- start usage -->

## Usage

to view the help message:

```
python -u run.py --help
```

to run the model:

```
python -u run.py --args args
```

<!-- end usage -->

Visit [documentation](documentation.html) to learn more about the `run.py` arguments.

Alternatively, check [Data Preprocessing](Tutorials/Data Prepocessing.ipynb), [Train, Test and Predict](Tutorials/Train,
Test and Predict.ipynb) and [Visualization](Tutorials/Testing Result Visualization.ipynb) notebooks in the `/Tutorials`
folder for more examples.

## Future Improvements

- Implement transfer learning to enhance inference speed and potentially gain insights into model interpretation.
- Utilize linear interpolation to augment the number of data samples.
- Enhance data preprocessing techniques:
    - Apply sliding window for further denoising.
    - Experiment with and potentially combine different data scalers.
- Incorporate the characteristic of power threshold in the model input, and if possible,
  use a combination of hinge loss and mean squared error (MSE) as the loss function.


## Contact

If you have any questions or suggestions, feel free to contact y.runyang@outlook.com

or describe it in Issues.

## Credits

- [Original TimesNet model](https://arxiv.org/abs/2210.02186)
- [Time-Series-Library](https://github.com/thuml/Time-Series-Library)