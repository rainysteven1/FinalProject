from typing import Dict, Tuple
from logging import Logger
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import json
import numpy as np
import os
import pandas as pd


class Analysis:
    ALGORITHMS = {"SVM": SVC, "LR": LogisticRegression, "RF": RandomForestClassifier}

    COLUMNS = ["sentiment"]

    SCALER = StandardScaler()

    def __init__(
        self,
        data_dir: str,
        workspace: str,
        logger: Logger,
        embedding_prefix: str,
        pca_dim: int,
        algorithm: str,
        param_grid: Dict,
    ):
        self.data_dir = data_dir
        self.workspace = workspace
        self.logger = logger
        self.embedding_prefix = embedding_prefix
        self.algorithm = algorithm
        self.param_grid = param_grid

        self.pca = PCA(n_components=pca_dim)

        self.logger.info(f"Embedding Prefix: {self.embedding_prefix}")
        self.logger.info(f"Param_Grid: {self.param_grid}")

    def _read_data(self, mode: str) -> Tuple[np.ndarray, pd.Series]:
        df = pd.read_csv(
            os.path.join(self.data_dir, f"{mode}.csv"), usecols=self.COLUMNS
        )

        x = np.load(
            os.path.join(self.workspace, f"{self.embedding_prefix}_{mode}.npy"),
            allow_pickle=True,
        )
        x = self.SCALER.fit_transform(x)
        x = self.pca.fit_transform(x)

        y = df["sentiment"]

        return x, y

    def classify(self) -> None:
        prefix = f"{self.embedding_prefix}_{self.algorithm}"

        X_train, y_train = self._read_data("train")
        X_test, y_test = self._read_data("test")

        model = self.ALGORITHMS[self.algorithm]()
        clf = GridSearchCV(
            model,
            self.param_grid,
            scoring=["accuracy", "f1", "precision", "roc_auc"],
            refit="accuracy",
            n_jobs=8,
            cv=2,
        )

        clf.fit(X_train, y_train)

        cv_results = clf.cv_results_
        df_results = pd.DataFrame(cv_results)
        df_results.to_csv(
            os.path.join(
                self.workspace, f"{self.embedding_prefix}_{self.algorithm}.csv"
            ),
            index=False,
        )
        with open(
            os.path.join(
                self.workspace, f"{self.embedding_prefix}_{self.algorithm}.json"
            ),
            "w",
        ) as f:
            json.dump(cv_results, f)

        self.logger.info(f"Best params: {clf.best_params_}")
        self.logger.info(f"Best score: {clf.best_score_}")

        best_model = clf.best_estimator_
        y_pred = best_model.predict(X_test)
        self.logger.info(f"Best Model Report: {classification_report(y_test, y_pred)}")

        report = classification_report(y_test, y_pred, output_dict=True)
        best_result_path = os.path.join(self.workspace, f"{prefix}.json")
        with open(best_result_path, "w") as f:
            json.dump(cv_results, f)

        df = pd.DataFrame(report)
        df.to_csv(os.path.join(self.workspace, "report.csv"), mode="a", index=False)
