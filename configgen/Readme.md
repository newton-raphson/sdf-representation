<!-- README FILE ACCORDING TO THE CONFIG_READER FILE -->
# Configuration Reader
This Python module, config_reader.py, is responsible for reading a configuration file (config.ini by default) and creating a Configuration object that contains all the parameters needed for training a model.

# Usage
```python 
from config_reader import Configuration 
config = Configuration('path_to_your_config.ini')
```

# Configuration Object Overview

The `Configuration` object is a crucial element of this application, housing various attributes that control the behavior and settings of the program. It relies on a structured `config.ini` file for initialization, and users must adhere to the specified format to ensure proper configuration parsing.

## Attributes

- **geometry:** File path to the geometry file.
- **directory:** Directory path for the files.
- **model:** The model to be used for training.
- **hidden_dim:** The dimension of the hidden layer in the model.
- **num_hidden_layers:** The number of hidden layers in the model.
- **input_dim:** The dimension of the input layer in the model.
- **skip_connection:** Determines if skip connections are used in the model.
- **beta:** The beta parameter for the model.
- **geometric_init:** Determines if geometric initialization is used in the model.
- **loss:** The loss function to be used for training.
- **lr:** The learning rate for training.
- **epochs:** The number of epochs for training.
- **minepochs:** The minimum number of epochs for training.
- **batchsize:** The batch size for training.
- **uniform_points:** The number of uniform points for sampling.
- **surface:** The surface parameter for sampling.
- **narrowband:** The narrowband parameter for sampling.
- **narrowband_width:** The width of the narrowband for sampling.
- **mismatchuse:** Determines if mismatch use is enabled for sampling.
- **train_test_split:** The ratio for splitting the data into training and testing sets.
- **ppo:** Determines if only post-processing is done.

## Loss Function

The `get_loss_function` method dynamically instantiates the loss function specified in the configuration file. The loss function must be defined in the `models` module. Parameters for the loss function are read from the configuration file.

## Note

The `Configuration` object is tailored to work with a specific structure of the `config.ini` file. Please ensure your configuration file adheres to the documented structure for correct parsing and proper functioning of the application. Refer to the provided sample `config.ini` file for guidance on the required format.
