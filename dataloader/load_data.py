import torch
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from utils.files import create_directory
from utils.constants import RANDOM_SEED_TEST_SPLIT
import numpy as np


def load_data(data_path, config, device):
    """
    Load data from CSV files and prepare it for training.

    Args:
        data_path (str): The path to the data directory.
        config (Config): The configuration object.
        device (torch.device): The device to load the data onto.

    Returns:
        training_dataloader (torch.utils.data.DataLoader): The data loader for training.
        validation_dataset (torch.utils.data.TensorDataset): The dataset for validation.
    """
    # check if the csv exists in the path or not
    df_uniform_points = pd.read_csv(os.path.join(save_directory, "uniform.csv"))
    df_on_surface = pd.read_csv(os.path.join(save_directory, "surface.csv"))
    df_narrow_band = pd.read_csv(os.path.join(save_directory, "narrow.csv"))
    columns = df_uniform_points.columns
    df_additional = pd.DataFrame(columns=columns)
    if config.mismatchuse:
        df_additional = pd.read_csv(os.path.join(data_path, "mismatch.csv"))

    # Create a list of data frames to concatenate, subject to the condition
    dfs_to_concat = [df for df in [df_uniform_points, df_on_surface, df_narrow_band, df_additional] if len(df) > 1]


    # Concatenate the data frames in the list if there are more than one
    df = pd.concat(dfs_to_concat, ignore_index=True)
    total_points = len(df)
    feature_columns = ['x', 'y', 'z']
    target_column = ['S', 'nx', 'ny', 'nz']


    X_train, val_X, y_train, val_Y = train_test_split(df[feature_columns], df[target_column], config.train_test_split, random_state=RANDOM_SEED_TEST_SPLIT)

    X = torch.tensor(X_train.values, dtype=torch.float32).to(device)
    Y = torch.tensor(y_train.values, dtype=torch.float32).to(device)
    val_X = torch.tensor(val_X[0].values, dtype=torch.float32).to(device)
    val_Y = torch.tensor(val_Y[0].values, dtype=torch.float32).to(device)

    training_dataset = torch.utils.data.TensorDataset(X, Y)
    training_dataloader = torch.utils.data.DataLoader(training_dataset, batch_size=config.batch_size, shuffle=True)
    validation_dataset = torch.utils.data.TensorDataset(val_X, val_Y)
    validation_dataloader = torch.utils.data.DataLoader(validation_dataset, batch_size=config.batch_size, shuffle=True)

    return training_dataloader, validation_dataloader
