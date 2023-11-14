import pathlib
import matplotlib.pyplot as plt





def get_files_from_dir_with_pathlib(dir_name):


    return list(pathlib.Path(dir_name).iterdir())



def get_accuracy( y_true , y_prob):

    y_prob = y_prob > 0.5
    return (y_true == y_prob).sum().item()



def plot(train, val, type="loss"):
    x = range(1, len(train)+1, 10)
    plt.plot(train, label="train")
    plt.plot(val, label="val")
    plt.xlabel("epochs")
    plt.ylabel(type)
    plt.xticks(x)
    plt.legend(loc= "upper right")
    plt.title(f"Training/validation {type}")
    plt.savefig(f"figures/{type}.png")
    plt.close()
