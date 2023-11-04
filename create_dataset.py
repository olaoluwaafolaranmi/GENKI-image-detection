import pandas as pd
from os import listdir
from os.path import join
from sklearn.model_selection import train_test_split
import json




def split_data(img_dir, label_dir, save_dir):

    cols = ["smile", "yaw", "pitch", "roll"]
    df = pd.read_csv(label_dir, sep=" ", header=None)
    df.columns = cols

    filenames = [join(img_dir,f) for f in listdir(img_dir)]
    df['image'] = filenames

    train, test = train_test_split(df, test_size=0.2, random_state=20)
    train, val = train_test_split(train, test_size=0.1, random_state=20)
    train_save_path = join(save_dir, "train.json")
    test_save_path = join(save_dir ,"test.json")
    val_save_path = join(save_dir,"val.json")
    
    save_json(train, train_save_path)
    save_json(test, test_save_path)
    save_json(val,val_save_path)

    return



def save_json(df,save_dir):

    new_json = df.to_json(orient="records")

    parsed = json.loads(new_json)

    json_str = json.dumps(parsed, indent=3)

    with open(save_dir,"w") as outfile:
        outfile.write(json_str)

    return


if __name__ == '__main__':

    img_dir = r"data\files"
    label_dir = r"data\labels.txt"
    save_dir = r"data\json_files"

    split_data(img_dir, label_dir, save_dir)





    
    