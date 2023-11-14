from pathlib import Path
import os
from typing import Union
import torch

data_dir = Path('data/GENKI-R2009a/Subsets/GENKI-4K/files')
label_list_file = Path('data/GENKI-R2009a/Subsets/GENKI-4K/GENKI-4K_Labels.txt')

pickle_data_path = Path('pickle_data_2')
best_early_stopping_model_path = Path('best_early_stopping_model')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_files_from_dir_with_pathlib(dir_name: Union[str, Path]) -> 'list[Path]':
    """Returns the files in the directory `dir_name` using the pathlib package.

    :param dir_name: The name of the directory.
    :type dir_name: str
    :return: The filenames of the files in the directory `dir_name`.
    :rtype: list[Path]
    """
    return sorted(list(Path(dir_name).iterdir()))


def empty_dir(dir_path: Path):
    """Erase all the files inside the directory.

    :param dir_path: Path of the directory.
    :type dir_path: Path
    """
    filepaths = get_files_from_dir_with_pathlib(dir_path)
    for f in filepaths:
        os.remove(f)