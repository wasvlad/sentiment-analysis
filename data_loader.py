import kagglehub
import os

os.environ['KAGGLEHUB_CACHE'] = os.path.join(os.getcwd(), "data")

# Download latest version
path = kagglehub.dataset_download("nelgiriyewithana/emotions")

print("Path to dataset files:", path)
