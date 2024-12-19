from logging import Logger
from logger import create_logger
import os
import shutil
import tarfile
import wget
import zipfile


def download_data(input_path: str, url: str, logger: Logger) -> str:
    if not os.path.exists(input_path):
        os.makedirs(input_path)

    logger.info(f"Downloading data from {url}")
    wget.download(url, out=input_path)
    logger.info(f"Downloaded data to {input_path}")

    file_name = url.split("/")[-1]
    file = os.path.join(input_path, file_name)

    if zipfile.is_zipfile(file):
        logger.info(f"Extracting zip file {file}")

        with zipfile.ZipFile(file, 'r') as f:
            file_names = f.namelist()
            f.extractall(input_path)

    if tarfile.is_tarfile(file):
        logger.info(f"Extracting tar file {file}")

        with tarfile.open(file, 'r') as f:
            file_names = f.getnames()
            f.extractall(input_path)

    shutil.rmtree(file, ignore_errors=True)
    file = os.path.join(input_path, file_names[0])
    logger.info(f"Extracted file is {file}")

    return file
