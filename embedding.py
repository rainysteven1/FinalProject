from logging import Logger
from gensim.models import KeyedVectors
import numpy as np
import os
import pandas as pd


class Embedding:
    MODEL_NAME = "GoogleNews-vectors-negative300.bin"

    def __init__(
        self,
        output_dir: str,
        logger: Logger,
        method: str = "Word2Vec",
        pretrained: bool = True,
    ):
        self.output_dir = output_dir
        self.logger = logger
        self.model = None

        if method == "Word2Vec":
            if pretrained:
                self.model = KeyedVectors.load_word2vec_format(
                    self.MODEL_NAME, binary=True
                )
                self.logger.info(
                    f"Loaded pretrained Word2Vec model: {os.path.basename(self.MODEL_NAME)[0]}"
                )

    def _build_vector(self, text: str):
        vec = np.zeros(self.model.vector_size).reshape((1, self.model.vector_size))
        count = 0

        for word in text:
            try:
                vec += self.model[word].reshape((1, self.model.vector_size))
                count += 1
            except KeyError:
                continue  # 如果词不在模型中，跳过

        if count != 0:
            vec /= count

        return vec

    def process(self, mode: str, df: pd.DataFrame) -> pd.DataFrame:
        self.logger.info("Start building vectors...")

        df["vector"] = df["content"].apply(lambda x: self._build_vector(x, self.model))
        df.to_csv(os.path.join(self.output_dir, f"embedded_{mode}.csv"), index=False)

        self.logger.info("Finish building vectors")

        return df
