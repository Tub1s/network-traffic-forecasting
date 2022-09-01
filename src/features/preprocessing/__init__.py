import os
import tqdm
import numpy as np
import pandas as pd

from collections import defaultdict
from typing import List, Tuple

def get_list_of_datapaths(path: str, sort: bool = True) -> List[str]:
    """
    Generate list of datafiles for given path.

    Args:
        path (str): Path to directory containing dataset
        sort (bool, optional): Sort datafiles in ascending order. Defaults to True.

    Returns:
        List[str]: List of datafiles for given path
    """
    list_of_datafiles = os.listdir(path)

    if sort:
        list_of_datafiles = sorted([int(x.replace('.txt', '')) for x in list_of_datafiles])
        list_of_datafiles = [path + str(file) + '.txt' for file in list_of_datafiles]

    else:
        list_of_datafiles = [path + file for file in list_of_datafiles]

    return list_of_datafiles

def load_data(datafiles: List[str], targets: List[Tuple[str, str]]) -> pd.DataFrame:
    """
    Function to load input data from data paths.

    Args:
        datafiles (List[str]): List of files to load
        targets (List[Tuple[str, str]]): List of pairs of nodes to load
    Returns:
        pd.DataFrame: Contains data on network traffic, value at given point in time n, for pairs of nodes
    """

    df = defaultdict(list)


    for file in tqdm.tqdm(datafiles):
        data = np.loadtxt(file)
        for pair in targets:
            df[f"{pair[0]}->{pair[1]}"].append(data[pair[0]][pair[1]])

    df = pd.DataFrame.from_dict(df)

    return df

def make_train_test_datasets(df: pd.DataFrame, split_point: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Creates split train/test split of DataFrame for stream of data.
    Train split contains data points before `split_point` and test split right after.

    Args:
        data (pd.DataFrame): Previously created DataFrame that contains sequential data
        split_point (int): Index of datapoint at which train/test split will be created

    Returns:
        pd.DataFrame: Train and test datasets
    """    
    train_dataset = df[:split_point]
    test_dataset = df[split_point:].reset_index().drop('index', axis=1)

    return train_dataset, test_dataset

def split_sequence(sequence: pd.Series, n_input_steps: int = 5, n_output_steps: int = 1) -> np.ndarray:
    """
    Splits sequential dataset into subsequences using sliding window.
    Creates pairs of input-output arrays containing subsequences of desired lengths. 
    Allows for experiments on both short and long term predicition windows.
    Example: for every 5 input steps, predict value in the next 5 steps.
        

    Args:
        sequence (pd.Series): Original sequential data
        n_input_steps (int, optional): Length of input subsequence. Defaults to 5.
        n_output_steps (int, optional): Length of output subsequence. Defaults to 1.

    Returns:
        np.ndarray: _description_
    """    
    X, y = list(), list()
    for i in tqdm.tqdm(range(sequence.shape[0])):
        if i + n_input_steps + n_output_steps < len(sequence) + 1:
            seq_x, seq_y = list(sequence[i:i+n_input_steps]), list(sequence[i+n_input_steps:i+n_input_steps+n_output_steps])
            X.append(seq_x)
            y.append(seq_y)

    return np.array(X), np.array(y)

