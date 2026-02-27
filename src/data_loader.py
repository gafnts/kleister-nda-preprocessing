import pandas as pd

from pathlib import Path
from typing import Literal


class DataLoader:
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self._column_names = pd.read_csv(
            data_dir / "in-header.tsv", sep="\t", encoding="utf-8", nrows=0
        ).columns.tolist()

    def load(
        self, partition: Literal["train", "dev-0", "test-A"] = "train"
    ) -> pd.DataFrame:
        if partition == "test-A":
            return self._read_data(partition)
        return pd.concat(
            [self._read_data(partition), self._read_labels(partition)], axis=1
        )

    def _read_data(self, partition: str) -> pd.DataFrame:
        return pd.read_csv(
            self.data_dir / partition / "in.tsv.xz",
            sep="\t",
            encoding="utf-8",
            compression="xz",
            header=None,
            names=self._column_names,
        )

    def _read_labels(self, partition: str) -> pd.DataFrame:
        return pd.read_csv(
            self.data_dir / partition / "expected.tsv",
            sep="\t",
            encoding="utf-8",
            header=None,
            names=["labels"],
        )
