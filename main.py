from datetime import datetime
from download import download_data
from logger import create_logger
from preprocess import DatasetPreprocessor
import os


if __name__ == "__main__":
    url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"

    now = datetime.now()
    formatted_now = now.strftime("%Y-%m-%d-%H-%M-%S")
    working_path = os.path.join("./output", formatted_now)
    if not os.path.exists(working_path):
        os.makedirs(working_path)

    logger = create_logger(working_path)

    input_dir = download_data("./input", url, logger)
    stopwords_path = "./resources/stopwords.txt"

    dp = DatasetPreprocessor(input_dir, working_path, stopwords_path, logger)
    dp.process_datasets()
