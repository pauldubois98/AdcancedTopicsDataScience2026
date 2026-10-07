# Download & un-zip the pizza / steak / sushi dataset (Food-3)
import urllib.request
import zipfile
from pathlib import Path

FILE_ID = "1JNiqVEbaOyRIWLc3UwHS4Y6CO60wx1iu"
URL = f"https://drive.usercontent.google.com/download?id={FILE_ID}&export=download&confirm=t"

data_path = Path("../data/")
zip_path = data_path / "Food-3.zip"

data_path.mkdir(parents=True, exist_ok=True)

if (data_path / "Food-3").is_dir():
    print(f"{data_path / 'Food-3'} already exists, skipping download.")
else:
    print("Downloading Food-3.zip (~160 MB)...")
    urllib.request.urlretrieve(URL, zip_path)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(data_path)
    zip_path.unlink()
    print(f"Dataset extracted to {(data_path / 'Food-3').resolve()}")
