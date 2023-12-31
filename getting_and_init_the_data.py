from pathlib import Path
from typing import Optional, Union
from dataset_class import Genki4KDataset
from torch.utils.data import DataLoader, random_split
import torch



def get_dataset(data_split: str,
                #data_dir: Union[str, Path],
                load_into_memory: Optional[bool] = True) \
        -> Genki4KDataset:
    return Genki4KDataset(
                    data_split=data_split,
                    #data_dir=data_dir,
                    load_into_memory=load_into_memory)


def get_data_loader(dataset: Genki4KDataset,
                    batch_size: int,
                    shuffle: bool,
                    drop_last: bool) \
        -> DataLoader:
    """Creates and returns a data loader.

    :param dataset: Dataset to use.
    :type dataset: dataset_class.MyDataset
    :param batch_size: Batch size to use.
    :type batch_size: int
    :param shuffle: Shuffle the data?
    :type shuffle: bool
    :return: Data loader, using the specified dataset.
    :rtype: torch.utils.data.DataLoader
    """
    return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, 
                      drop_last=drop_last, num_workers=1, collate_fn=collate_fn)


def get_all_data_loaders(batch_size: int):
    train_dataset = get_dataset('train')
    test_dataset = get_dataset('test')
    val_dataset = get_dataset('val')

    # Create the data loaders
    train_loader = get_data_loader(train_dataset, batch_size, True, False)
    test_loader = get_data_loader(test_dataset, batch_size, False, False)
    val_loader = get_data_loader(val_dataset, batch_size, False, False)

    return train_loader,val_loader, test_loader


def collate_fn(batch):
  
  return torch.stack([x for x, _ in batch], dim=0), torch.tensor([y for _,y in batch])



def main():
    pass


if __name__ == "__main__":
    main()