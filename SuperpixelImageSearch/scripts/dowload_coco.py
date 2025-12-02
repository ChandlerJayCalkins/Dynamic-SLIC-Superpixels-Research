import os
import zipfile
from urllib.request import urlretrieve

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA = os.path.join(ROOT, "data", "coco2017")
IMG = os.path.join(DATA, "images")
ANN = os.path.join(DATA, "annotations")

URLS = {
    "train2017.zip": "http://images.cocodataset.org/zips/train2017.zip",
    "val2017.zip": "http://images.cocodataset.org/zips/val2017.zip",
    "annotations_trainval2017.zip": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
}

def download(url, dst):
    print(f"Downloading {url} → {dst} ...")
    urlretrieve(url, dst)

def unzip(path, dst):
    print(f"Unzipping {path} → {dst}")
    with zipfile.ZipFile(path, "r") as zf:
        zf.extractall(dst)

def main():
    os.makedirs(IMG, exist_ok=True)
    os.makedirs(ANN, exist_ok=True)

    # Download all files
    for filename, url in URLS.items():
        dst = os.path.join(DATA, filename)
        if not os.path.exists(dst):
            download(url, dst)
        else:
            print(f"{filename} already exists.")

    # Extract
    unzip(os.path.join(DATA, "train2017.zip"), IMG)
    unzip(os.path.join(DATA, "val2017.zip"), IMG)
    unzip(os.path.join(DATA, "annotations_trainval2017.zip"), ANN)

    print("COCO dataset download complete!")

if __name__ == "__main__":
    main()
