#!/usr/bin/env python
from IPython import get_ipython
import zipfile
import os


def remove(filename):
    if os.path.exists(filename):
        os.remove(filename)


def unzip(filename):
    with zipfile.ZipFile(filename, "r") as zip_ref:
        zip_ref.extractall()


if __name__ == "__main__":
    try:
        import kaggle
    except ImportError:
        print("Kaggle API not installed, installing now")
        get_ipython().system("pip install kaggle")

    try:
        print("Data found, extracting now")
        unzip("data.zip")
        remove("data.zip")

    except FileNotFoundError:
        print("Data not found, downloading now")
        get_ipython().system('echo "Downloading expedia data"')
        get_ipython().system(
            "kaggle competitions download -c expedia-personalized-sort -f data.zip"
        )
        get_ipython().system('echo "This is working"')

        unzip("data.zip")
        remove("data.zip")
