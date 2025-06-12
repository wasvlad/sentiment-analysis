import kagglehub
import os
import shutil

# Download latest version
path = kagglehub.dataset_download("nelgiriyewithana/emotions")

# Move the data file to ./data/raw_data.csv
src_file = os.path.join(path, "text.csv")
dst_file = os.path.join(os.getcwd(), "data", "raw_data.csv")
os.makedirs(os.path.dirname(dst_file), exist_ok=True)
shutil.copy2(src_file, dst_file)

print("Path to dataset files:", path)
print("Raw data copied to:", dst_file)
