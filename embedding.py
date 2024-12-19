from typing import Dict, Union
from logging import Logger
from gensim.models import KeyedVectors, Word2Vec
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
        model_params: Union[str, Dict],
    ):
        self.data_dir = data_dir
        self.workspace = workspace
        self.logger = logger
        self.num = num
        self.method = method
        self.model_params = model_params

        self.embedding_prefix = None

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

    def _impl_model(self, text: str) -> np.ndarray:
        doc = [word for word in text if word in self.model.wv]
        return (
            np.mean(self.model.wv[doc], axis=0)
            if len(doc) > 0
            else np.zeros(self.model.vector_size)
        )

    def _process_data_word2vec(self, data_path: str) -> np.ndarray:
        if isinstance(self.model_params, str):
            self.model = KeyedVectors.load_word2vec_format(
                self.model_params, binary=True
            )
            self.embedding_prefix = f"{self.method}_pretrained"

            datas = list()
            for chunk in pd.read_csv(
                data_path, chunksize=self.CHUNK_SIZE, usecols=self.COLUMNS
            ):
                datas.append(
                    chunk[self.COLUMNS[0]]
                    .apply(ast.literal_eval)
                    .apply(self._build_vector)
                )

            return np.vstack(pd.concat(datas))

        else:
            df = pd.read_csv(data_path, usecols=self.COLUMNS)
            self.model = Word2Vec(df[self.COLUMNS[0]], **self.model_params)

            model_name = "cbow" if self.model_params["sg"] == 0 else "skipgram"
            self.embedding_prefix = f"{self.method}_{model_name}"

            datas = df[self.COLUMNS[0]].apply(ast.literal_eval).apply(self._impl_model)
            datas = datas.to_list()

            return np.vstack(datas)

    def _process_data(self, mode: str) -> None:
        self.logger.info(f"Start building {mode} vectors...")

        if self.method == "word2vec":
            func = self._process_data_word2vec

        data_path = os.path.join(self.data_dir, f"{mode}.csv")
        self.progress_bar = tqdm(total=self.num, desc=f"Processing {mode}")

        arr = func(data_path)
        npy_path = os.path.join(self.workspace, f"{self.embedding_prefix}_{mode}.npy")

        np.save(npy_path, arr)

        self.logger.info(f"Save vectors to {npy_path}")

    def process(self) -> str:
        self._process_data("train")
        self._process_data("test")

        return self.embedding_prefix
