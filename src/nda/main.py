import logging
import pandas as pd

from typing import List
from pathlib import Path

from nda.data_loader import DataLoader, Partition
from nda.label_transformer import LabelTransformer


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

logger = logging.getLogger(__name__)


DATA_DIR: Path = Path(__file__).parent / "static" / "data"
OUTPUT_DIR: Path = Path(__file__).parent / "static" / "outputs"
PARTITIONS: tuple[Partition, Partition, Partition] = ("train", "dev-0", "test-A")


def load_data() -> List[pd.DataFrame]:
    logger.info("Loading data for partitions: %s", PARTITIONS)
    loader = DataLoader(DATA_DIR)
    dataframes = [loader.load(partition) for partition in PARTITIONS]
    logger.info("Loaded dataframes: %s", [df.shape for df in dataframes])
    return dataframes


def parse_labels(
    dataframes: List[pd.DataFrame],
) -> List[pd.DataFrame]:
    logger.info("Parsing labels for dataframes")
    transformed = [
        LabelTransformer.transform(df, partition)
        for df, partition in zip(dataframes, PARTITIONS)
    ]
    logger.info("Labels parsed for all partitions")
    return transformed


def store_parquet(dataframes: List[pd.DataFrame]) -> None:
    for df, partition in zip(dataframes, PARTITIONS):
        logger.info("Storing parquet for partition: %s, shape: %s", partition, df.shape)
        LabelTransformer.to_parquet(df, OUTPUT_DIR / partition)
    logger.info("All partitions stored as parquet")


def main() -> None:
    logger.info("Starting main pipeline")
    train, val, test = load_data()
    train, val, test = parse_labels([train, val, test])
    store_parquet([train, val, test])
    logger.info("Pipeline completed")


if __name__ == "__main__":
    main()
