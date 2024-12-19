from logging import Logger
from gensim.models import KeyedVectors
import ast
import numpy as np
import os
import pandas as pd
from tqdm import tqdm


class TextVectorizer:
    CHUNK_SIZE = 100000

    COLUMNS = ["content"]

    def __init__(
        self,
        data_dir: str,
        workspace: str,
        logger: Logger,
        num: int,
        method: str,
        model_name: str,
    ):
        self.data_dir = data_dir
        self.workspace = workspace
        self.logger = logger
        self.num = num
        self.method = method
        self.model = None

        if method == "word2vec":
            self.model = KeyedVectors.load_word2vec_format(model_name, binary=True)

    def _build_vector(self, text: str):
        vec = np.zeros(self.model.vector_size).reshape((1, self.model.vector_size))
        count = 0

        for word in text:
            try:
                vec += self.model[word].reshape((1, self.model.vector_size))
                count += 1
            except KeyError:
                continue 

        if count != 0:
            vec /= count

        self.progress_bar.update(1)

        return vec

    def _process_data(self, mode: str) -> None:
        self.logger.info(f"Start building {mode} vectors...")

        data_path = os.path.join(self.data_dir, f"{mode}.csv")
        datas = list()
        self.progress_bar = tqdm(total=self.num, desc=f"Processing {mode}")

        for chunk in pd.read_csv(
            data_path, chunksize=self.CHUNK_SIZE, usecols=self.COLUMNS
        ):
            datas.append(
                chunk["content"].apply(ast.literal_eval).apply(self._build_vector)
            )

        arr = np.vstack(pd.concat(datas))
        npy_path = os.path.join(self.workspace, f"{self.method}_{mode}.npy")
        np.save(npy_path, arr)

        self.logger.info(f"Save vectors to {npy_path}")

    def process(self):
        self._process_data("train")
        self._process_data("test")
