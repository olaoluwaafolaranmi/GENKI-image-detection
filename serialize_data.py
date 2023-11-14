import pickle
from pathlib import Path
from PIL import Image
import re
import numpy as np
import random
import pandas as pd

from utils import data_dir, label_list_file, pickle_data_path, get_files_from_dir_with_pathlib, empty_dir


def main():
    empty_dir(Path(pickle_data_path, 'train'))
    empty_dir(Path(pickle_data_path, 'test'))
    image_list = get_files_from_dir_with_pathlib(data_dir)

    with open(label_list_file, 'r') as f:
        label_list = f.readlines()

    label_list = [int(label.strip().split(' ')[0]) for label in label_list]

    # 80/20 for train and val
    df = {
        'file_path': image_list,
        'label': label_list,
        'split': []
    }
    split = ['train'] * int(len(label_list)*0.6) + ['val']* int(len(label_list)*0.2) + ['test'] * int(len(label_list)*0.2)
    random.shuffle(split)
    df['split'] = split
    df = pd.DataFrame(df)

    for row in df.iterrows():
        img_path = row[1]['file_path']
        img_arr = np.asarray(Image.open(img_path).resize((64,64)))
        
        if img_arr.ndim == 2:
            img_arr = np.expand_dims(img_arr, axis=2)
            img_arr = np.repeat(img_arr, 3, axis=2)
            print(img_arr.shape)

        img_arr = img_arr.transpose([2,0,1])
        label = row[1]['label']

        data = {'img': img_arr, 'label': label}
        with open(Path(pickle_data_path, row[1]['split'], img_path.stem + '.pkl'), 'wb') as f:
            pickle.dump(data, f)


if __name__ == '__main__':
    main()