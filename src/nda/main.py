import pandas as pd

from pathlib import Path
from typing import List
from nda.data_loader import DataLoader, Partition


DATA_DIR: Path = Path(__file__).parent / "static" / "data"
OUTPUT_DIR: Path = Path(__file__).parent / "static" / "outputs"
PARTITIONS: tuple[Partition, Partition, Partition] = ("train", "dev-0", "test-A")


def load_data() -> List[pd.DataFrame]:
    loader = DataLoader(
        data_dir=DATA_DIR,
        output_dir=OUTPUT_DIR,
    )

    return [loader.load(partition) for partition in PARTITIONS]


def jsonify_labels() -> None:
    pass


def main() -> None:
    train, val, test = load_data()

    print(train.shape)
    print(val.shape)
    print(test.shape)


if __name__ == "__main__":
    main()
