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
**Files**
- **geometry:** File path to the geometry file, folder path if distributed
- **directory:** Directory path for the files.
- **name:** Name of the folder where all the files  are saved

**Model**

- **model:** Name of the model must be in the ```./model/networks.py```
- **hidden_dim:** The dimension of the hidden layer in the model.
- **num_hidden_layers:** The number of hidden layers in the model.
- **input_dim:** The dimension of the input layer in the model.
- **skip_connection:** Determines if skip connections are used in the model.
- **beta:** The beta parameter for the model.(see ImplicitNet)
- **geometric_init:** Determines if geometric initialization is used in the model.

**Loss**

- **loss:** The loss function to be used for training.
- **k-v pair** The key value pairs required for the loss function

**Training**  

- **lr:** The learning rate for training.
- **epochs:** The number of epochs for training.
- **minepochs:** The minimum number of epochs for training.
- **batchsize:** The batch size for training.
- **checkpointing:** The frequency of checkpointing
- **continue:** If the model is to be continued training from scratch
- **patience:** The minimum number of epochs to be trained

**Sampling**

- **samplingonly:** If set true exists after sampling
- **rescale:** If set true the geometry is rescaled between -1 and 1
- **distributed:** If set true distribute geometry is handled
- **uniform_points:** The number of uniform points for sampling.
- **surface:** The surface parameter for sampling.
- **narrowband:** The narrowband parameter for sampling.
- **narrowband_width:** The width of the narrowband for sampling.
- **mismatchuse:** Determines if mismatch use is enabled for sampling.
- **train_test_split:** The ratio for splitting the data into training and testing sets.

**Optional**

- **ppo:** Determines if only post-processing is done.
- **reconstruct:** If reconstruct set true exits after reconstruction
- **cubesize:** The cube dimension for reconstruction and postprocess
- **postprocessbatchsize:** The batch size according to the memory size

## Loss Function

The `get_loss_function` method dynamically instantiates the loss function specified in the configuration file. The loss function must be defined in the `model/losses` module. Parameters for the loss function are read from the configuration file.

## Note

The `Configuration` object is tailored to work with a specific application in mind. Please ensure your configuration file adheres to the documented structure for correct parsing and proper functioning of the application. Refer to the provided sample `tests/test.ini` file for guidance on the required format.
