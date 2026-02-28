import pandas as pd

from pathlib import Path
from typing import List, Tuple

from nda.data_loader import DataLoader, Partition
from nda.label_converter import LabelConverter


DATA_DIR: Path = Path(__file__).parent / "static" / "data"
OUTPUT_DIR: Path = Path(__file__).parent / "static" / "outputs"
PARTITIONS: tuple[Partition, Partition, Partition] = ("train", "dev-0", "test-A")


def load_data() -> List[pd.DataFrame]:
    loader = DataLoader(DATA_DIR)
    return [loader.load(partition) for partition in PARTITIONS]


def parse_labels(dataframes: Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]) -> None:
    for partition, df in zip(PARTITIONS, dataframes):
        df = LabelConverter(df).convert_labels(partition)
        df.to_parquet(OUTPUT_DIR / f"{partition}_converted.parquet", index=False)


def main() -> None:
    train, val, test = load_data()
    parse_labels((train, val, test))

    print(train.shape)
    print(val.shape)
    print(test.shape)


if __name__ == "__main__":
    main()
