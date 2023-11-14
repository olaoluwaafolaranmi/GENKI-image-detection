from typing import Tuple, Optional, Union, Dict
from pickle import load as pickle_load
from pathlib import Path
import json
from PIL import Image
import torchvision.transforms as transforms


from torch.utils.data import Dataset
import numpy
import torch

from utils import get_files_from_dir_with_pathlib


class Genki4KDataset(Dataset):
    def __init__(self,
                 data_split: str,
                 #data_dir: Union[str, Path],
                 load_into_memory: Optional[bool] = True) \
            -> None:
        """An example of an object of class torch.utils.data.Dataset

        :param data_dir: Directory to read data from.
        :type data_dir: str
        :param data_parent_dir: Parent directory of the data, defaults\
                                to ``.
        :type data_parent_dir: str
        :param key_features: Key to use for getting the features,\
                             defaults to `features`.
        :type key_features: str
        :param key_class: Key to use for getting the class, defaults\
                          to `class`.
        :type key_class: str
        :param load_into_memory: Load the data into memory? Default to True
        :type load_into_memory: bool
        """
        super().__init__()
        json_path = f"data/json_files/{data_split}.json"

        with open(json_path, 'r') as f:
            json_obj = json.load(f)


        self.img_paths = [item["image"] for item in json_obj]
        self.labels = [item["smile"] for item in json_obj]
        # data_path = Path(data_dir, data_split)
        # available_files = get_files_from_dir_with_pathlib(data_path)
        self.files = []

        # for i in range(len(available_files)):
        #     if i%4 == 0:
        #         self.files.append(available_files[i])

        
        self.load_into_memory = load_into_memory
        self.key_img = 'img'
        self.key_label = 'label'
        if self.load_into_memory:
            for i, img_path in enumerate(self.img_paths):
                file_dict = {}
                file_dict[self.key_img] = self._load_image(img_path)
                file_dict[self.key_label] = self.labels[i]
                self.files.append(file_dict)

    @staticmethod
    def _load_image(file_path: Path):
        """Loads a file using pathlib.Path

        :param file_path: File path.
        :type file_path: pathlib.Path
        :return: The file.
        :rtype: dict[str, int|numpy.ndarray]
        """

        img = Image.open(file_path)
        no_channels = len(img.getbands())
        if no_channels != 3:
            img = img.convert("RGB")
        img_resized = img.resize((64,64))

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        img_tensor = transform(img_resized)
        
        return img_tensor


    def __len__(self) \
            -> int:
        """Returns the number of speech/no_speech pair from the dataset.

        :return: Number of pairs in the dataset
        :rtype: int
        """
        return len(self.img_paths)

    def __getitem__(self,
                    item: int) \
            -> Tuple[torch.Tensor, int]:
        """Returns an item from the dataset.
        
        :param item: Index of the pair.
        :type item: int
        :return: Features and class of the item.
        :rtype: (numpy.ndarray, int)
        """
        if self.load_into_memory:
            the_item: Dict[str, Union[int, torch.Tensor]] = self.files[item]
        else:
            the_item = {}
            the_item[self.key_img] = self._load_image(self.img_paths[item])
            the_item[self.key_label] = self.labels[item]

        return the_item[self.key_img], the_item[self.key_label]
