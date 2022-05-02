import os
import tqdm
import numpy as np
import pandas as pd

from collections import defaultdict
from typing import List

def get_list_of_datafiles(path: str, sort: bool = True) -> List[str] | List[int]:
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

    return list_of_datafiles

def load_data(path: str, datafiles: List[str] | List[int]) -> defaultdict(list):
    df = defaultdict(list)

    if datafiles[0] is type(str):
        for file in tqdm.tqdm(datafiles):
            temp = np.loadtxt(f"{path}{file}")
            df['5->8'].append(temp[5][8])
            df['8->5'].append(temp[8][5])
            df['5->12'].append(temp[5][12])
            df['8->12'].append(temp[8][12])
    
    else: 
        for file in tqdm.tqdm(datafiles):
            temp = np.loadtxt(f"{path}{str(file)}.txt")
            #temp = np.loadtxt(f"{DATA_PATH}{file}")
            df['5->8'].append(temp[5][8])
            df['8->5'].append(temp[8][5])
            df['5->12'].append(temp[5][12])
            df['8->12'].append(temp[8][12])

    df = pd.DataFrame.from_dict(df)

    return df

def make_train_test_datasets(data: pd.DataFrame, split_point: int) -> pd.DataFrame:
    train_dataset = data[:split_point]
    test_dataset = data[split_point:].reset_index().drop('index', axis=1)

    return train_dataset, test_dataset

def split_sequence(sequence: pd.Series, n_input_steps: int = 5, n_output_steps: int = 1) -> np.ndarray:
    X, y = list(), list()
    for i in tqdm.tqdm(range(sequence.shape[0])):
        if i + n_input_steps + n_output_steps < len(sequence) + 1:
            seq_x, seq_y = list(sequence[i:i+n_input_steps]), list(sequence[i+n_input_steps:i+n_input_steps+n_output_steps])
            X.append(seq_x)
            y.append(seq_y)

    return np.array(X), np.array(y)

