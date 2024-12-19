from analyze import Analysis
from datetime import datetime
from download import download_data
from embedding import TextVectorizer
from logger import create_logger
from preprocess import DatasetPreprocessor
import argparse
import os
import json

CONFIG_PATH = "./resources/config.json"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sentiment Classification")
    parser.add_argument("-p", "--preprocess", type=bool, default=False)
    args = parser.parse_args()

    with open(CONFIG_PATH) as f:
        config = json.load(f)

    now = datetime.now()
    formatted_now = now.strftime("%Y-%m-%d-%H-%M-%S")
    data_dir = os.path.join(config["data_dir"], formatted_now)
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    workspace = os.path.join(config["output_dir"], formatted_now)
    if not os.path.exists(workspace):
        os.makedirs(workspace)

    logger = create_logger(workspace)

    if args.preprocess:
        input_dir = download_data(config["input_dir"], f["dataset"]["url"], logger)

        logger.info("-" * 15 + "Preprocessing" + "-" * 15)
        dp = DatasetPreprocessor(input_dir, data_dir, config["stopwords_path"], logger)
        dp.process_datasets()

    logger.info("-" * 15 + "Embedding" + "-" * 15)
    for method in config["embedding"]:
        vectorizer = TextVectorizer(
            data_dir, workspace, logger, f["dataset"]["num"], method
        )
        vectorizer.process("train")
        vectorizer.process("test")

    logger.info("-" * 15 + "Classification" + "-" * 15)
    for embedding_method in config["embedding"]:
        for param_grid in config["classification"]["configs"]:
            analyses = Analysis(
                workspace,
                embedding_method,
                config["classification"]["pca_dim"],
                param_grid,
            )
            analyses.classify()
