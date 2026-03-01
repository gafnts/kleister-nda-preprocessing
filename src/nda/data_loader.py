"""
DataLoader for reading compressed TSV input files and expected labels by partition.
"""

from pathlib import Path
from typing import Literal

import pandas as pd

Partition = Literal["train", "dev-0", "test-A"]


class DataLoader:
    """
    Loads input and label TSV files for a given dataset partition.
    """

    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self._column_names = pd.read_csv(
            data_dir / "in-header.tsv", sep="\t", encoding="utf-8", nrows=0
        ).columns.tolist()

    def load(self, partition: Partition = "train") -> pd.DataFrame:
        """
        Return the input dataframe for the partition, joined with labels when available.
        """
        if partition == "test-A":
            return self._read_data(partition)
        return pd.concat(
            [self._read_data(partition), self._read_labels(partition)], axis=1
        )

    def _read_data(self, partition: Partition) -> pd.DataFrame:
        """
        Read the xz-compressed input TSV for the given partition.
        """
        return pd.read_csv(
            self.data_dir / partition / "in.tsv.xz",
            sep="\t",
            encoding="utf-8",
            compression="xz",
            header=None,
            names=self._column_names,
        )

    def _read_labels(self, partition: Partition) -> pd.DataFrame:
        """
        Read the expected.tsv label file for the given partition.
        """
        return pd.read_csv(
            self.data_dir / partition / "expected.tsv",
            sep="\t",
            encoding="utf-8",
            header=None,
            names=["labels"],
        )
