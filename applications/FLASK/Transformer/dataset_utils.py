import argparse
import importlib
import os
import sys
from typing import List

dataset_dir = os.path.join(os.path.dirname(__file__), "datasets")


def available_datasets() -> List[str]:
    """
    Returns the available datasets in the dataset folder.
    """
    result = []
    for file in os.listdir(dataset_dir):
        if file.endswith(".py"):
            result.append(os.path.basename(file)[:-3])
    return result


def load_dataset(name: str):
    """
    Loads a dataset by importing the requested module.
    """
    sys.path.append(dataset_dir)
    return importlib.import_module(name)
