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

    with open(CONFIG_PATH, "r") as f:
        config = json.load(f)

    now = datetime.now()
    formatted_now = now.strftime("%Y-%m-%d-%H-%M-%S")
    data_dir = os.path.join(config["data_dir"])
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
    embedding_prefix_list = list()
    for params in config["embedding"]:
        vectorizer = TextVectorizer(
            data_dir, workspace, logger, config["dataset"]["num"], **params
        )
        embedding_prefix_list.append(vectorizer.process())

    logger.info("-" * 15 + "Classification" + "-" * 15)
    for embedding_prefix in embedding_prefix_list:
        for params in config["classification"]["configs"]:
            analyses = Analysis(
                data_dir,
                workspace,
                logger,
                embedding_prefix,
                config["classification"]["pca_dim"],
                **params,
            )
            analyses.classify()
