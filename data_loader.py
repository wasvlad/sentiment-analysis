import random
import kagglehub
import os
import shutil
import pandas as pd

def load_data(random_seed=42):
    """
    Downloads the dataset from Kaggle and saves it to the specified directory.
    """
    # Download latest version
    path = kagglehub.dataset_download("nelgiriyewithana/emotions")

    # Move the data file to ./data/raw_data.csv
    src_file = os.path.join(path, "text.csv")
    dst_file = os.path.join(os.getcwd(), "data", "raw_data.csv")
    os.makedirs(os.path.dirname(dst_file), exist_ok=True)
    shutil.copy2(src_file, dst_file)
    print("Path to dataset files:", path)
    print("Raw data copied to:", dst_file)
    data = pd.read_csv(dst_file)
    x = data['text'].to_list()
    y = data['label'].to_numpy()
    indexes = list(range(len(x)))
    random.seed(random_seed)  # For reproducibility
    random.shuffle(indexes)
    temp = []
    for i in indexes:
        temp.append(x[i])
    x = temp
    y = y[indexes]
    x_train = x[:int(len(x) * 0.6)]
    y_train = y[:int(len(y) * 0.6)]
    x_val = x[int(len(x) * 0.6):int(len(x) * 0.8)]
    y_val = y[int(len(y) * 0.6):int(len(y) * 0.8)]
    x_test = x[int(len(x) * 0.8):]
    y_test = y[int(len(y) * 0.8):]
    return x_train, y_train, x_val, y_val, x_test, y_test

if __name__ == "__main__":
    print(load_data())
    print("Data loading complete.")