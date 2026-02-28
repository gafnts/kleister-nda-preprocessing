import pandas as pd

from typing import List
from pathlib import Path

from nda.data_loader import DataLoader, Partition
from nda.label_transformer import LabelTransformer


DATA_DIR: Path = Path(__file__).parent / "static" / "data"
OUTPUT_DIR: Path = Path(__file__).parent / "static" / "outputs"
PARTITIONS: tuple[Partition, Partition, Partition] = ("train", "dev-0", "test-A")


def load_data() -> List[pd.DataFrame]:
    loader = DataLoader(DATA_DIR)
    return [loader.load(partition) for partition in PARTITIONS]


def parse_labels(
    dataframes: List[pd.DataFrame],
) -> List[pd.DataFrame]:
    return [
        LabelTransformer.transform(df, partition)
        for df, partition in zip(dataframes, PARTITIONS)
    ]


def store_parquet(dataframes: List[pd.DataFrame]) -> None:
    for df, partition in zip(dataframes, PARTITIONS):
        LabelTransformer.to_parquet(df, OUTPUT_DIR / partition)


def main() -> None:
    train, val, test = load_data()
    train, val, test = parse_labels([train, val, test])
    store_parquet([train, val, test])


if __name__ == "__main__":
    main()
