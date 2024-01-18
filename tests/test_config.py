import pytest
import os
from configgen.config_reader import Configuration
from model.networks import ImplicitNet

def test_configuration():
    # Create a Configuration object with a test config file
    config_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test_config.ini')
    config = Configuration(config_file_path)

    # Check that the values read from the config file are correct
    # files
    assert config.geometry == 'some_geometry_file'
    assert config.directory == 'some_directory'
    # models
    assert config.model.__class__.__name__ == 'ImplicitNet'
    assert config.hidden_dim == 256
    assert config.num_hidden_layers == 2
    assert config.input_dim == 3
    assert config.skip_connection == ()
    assert config.beta == 0
    assert config.geometric_init == False
    # loss
    assert config.loss.__class__.__name__ == 'WeightedSmoothL2Loss'
    # training
    assert config.lr == 0.001
    assert config.epochs == 100
    assert config.minepochs == 10
    assert config.batchsize == 64
    # sampling
    assert config.uniform_points == 1000
    assert config.surface == 500
    assert config.narrowband == 500
    assert config.narrowband_width == 0.1
    assert config.mismatchuse == False
    assert config.train_test_split == 0.8
    # optional
    assert config.ppo == False