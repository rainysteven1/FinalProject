from typing import List
from bs4 import BeautifulSoup
from functools import reduce
from logging import Logger
from tqdm import tqdm
import glob
import os
import pandas as pd
import re
import spacy

__all__ = ["DatasetPreprocessor"]


class _TextPreprocessor:
    PUNCTUATION = r"""!"#$%&()*+,-./:;<=>?@[\]^_`{|}~"""

    COLUMNS = ["text", "lemma", "pos", "tag"]

    STARTS = ("J", "V", "N", "R")

    SEPERATOR = " "

    MODEL = "en_core_web_sm"

    CONTRACTIONS = {
        r"(it|he|she|that|this|there|here)('s)": r"\1 is",
        r"([a-zA-Z])('s)": r"\1",
        r"([a-zA-Z])(n't)": r"\1 not",
        r"([a-zA-Z])('d)": r"\1 would",
        r"([a-zA-Z])('ll)": r"\1 will",
        r"([I|i])('m)": r"\1 am",
        r"([a-zA-Z])('re)": r"\1 are",
        r"([a-zA-Z])('ve)": r"\1 have",
    }

    def __init__(self, num: int, stopwords_path: str) -> None:
        self.translator = str.maketrans("", "", self.PUNCTUATION)
        self.pat_letter = re.compile(r"[^a-zA-Z \']+")
        self.patterns = [
            (re.compile(pattern, re.I), replacement)
            for pattern, replacement in self.CONTRACTIONS.items()
        ]
        self.nlp = spacy.load(self.MODEL)
        self.progress_bar = tqdm(total=num, desc="Processing")

        with open(stopwords_path, "r") as f:
            self.stopwords = set([line.strip() for line in f.readlines()])

    def _clean_html(self, text: str) -> str:
        return BeautifulSoup(text, "html.parser").get_text(separator=self.SEPERATOR)

    def _remove_punctuation(self, text: str) -> str:
        return text.translate(self.translator)

    def _expand_contractions(self, text: str) -> str:
        text = self.pat_letter.sub(self.SEPERATOR, text).strip()
        for pattern, replacement in self.patterns:
            text = pattern.sub(replacement, text)

        return text.replace("'", self.SEPERATOR)

    def _lemmazatiz(self, text: str) -> List[str]:
        temp = self.nlp(text)

        entities = list()
        for ent in temp.ents:
            entities.extend(ent.text.split(" "))

        exclude_words = self.stopwords | set(entities)

        token_details = list()
        for token in temp:
            token_details.append((token.text, token.lemma_, token.pos_, token.tag_))

        df_details = pd.DataFrame(token_details, columns=self.COLUMNS)
        filtered_df = df_details[
            df_details["tag"].str.startswith(self.STARTS)
            & ~df_details["text"].isin(exclude_words)
        ]
        filtered_df.loc[:, "lemma"] = filtered_df["lemma"].apply(lambda x: x.lower())

        return filtered_df["lemma"].tolist()

    def process_text(self, text: str) -> List[str]:
        pipeline = [
            self._clean_html,
            self._remove_punctuation,
            self._expand_contractions,
            self._lemmazatiz,
        ]
        tokens = reduce(lambda x, func: func(x), pipeline, text)

        self.progress_bar.update(1)

        return tokens


class DatasetPreprocessor:
    COLUMNS = ["index", "star", "content", "sentiment"]

    def __init__(
        self, input_dir: str, output_dir: str, stopwords_path: str, logger: Logger
    ) -> None:
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.stopwords_path = stopwords_path
        self.logger = logger

    def _load_sentiment_dataset(self, mode: str) -> pd.DataFrame:
        mode_dir = os.path.join(self.input_dir, mode)
        pos_dir = os.path.join(mode_dir, "pos")
        neg_dir = os.path.join(mode_dir, "neg")

        def get_sentiment_df(dir: str) -> pd.DataFrame:
            self.logger.info(f"Loading {dir} data")
            sentiment = 1 if os.path.split(dir)[-1] == "pos" else -1
            datas = list()

            for file in glob.glob(os.path.join(dir, "*.txt")):
                file_name = os.path.basename(file).split(".")[0]
                data = [*file_name.split("_"), sentiment]

                with open(file, "r") as f:
                    data.insert(-1, f.read())

                datas.append(data)

            return pd.DataFrame(datas, columns=self.COLUMNS)

        df_pos = get_sentiment_df(pos_dir)
        df_neg = get_sentiment_df(neg_dir)

        return pd.concat([df_pos, df_neg], ignore_index=True)

    def _process_dataset_partition(self, mode: str) -> None:
        self.logger.info(f"Start processing {mode} dataset...")

        df = self._load_sentiment_dataset(mode).head(5)
        text_preprocessor = _TextPreprocessor(df.shape[0], self.stopwords_path)

        df.loc[:, "content"] = df["content"].apply(text_preprocessor.process_text)

        df.to_csv(os.path.join(self.output_dir, f"{mode}.csv"), index=False)
        self.logger.info(f"Processed {mode} dataset")

    def process_datasets(self) -> None:
        self._process_dataset_partition("train")
        self._process_dataset_partition("test")
