import logging
from pathlib import Path

import pandas as pd

from nda.data_loader import DataLoader, Partition
from nda.document_relocator import DocumentRelocator
from nda.label_transformer import LabelTransformer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

logger = logging.getLogger(__name__)


DATA_DIR: Path = Path(__file__).parent / "static" / "data"
OUTPUT_DIR: Path = Path(__file__).parent / "static" / "outputs"
PARTITIONS: tuple[Partition, ...] = ("train", "dev-0", "test-A")


def load_data() -> list[pd.DataFrame]:
    logger.info("Loading data for partitions: %s", PARTITIONS)
    loader = DataLoader(DATA_DIR)
    dataframes = [loader.load(partition) for partition in PARTITIONS]
    logger.info("Loaded dataframes: %s", [df.shape for df in dataframes])
    return dataframes


def parse_labels(
    dataframes: list[pd.DataFrame],
) -> list[pd.DataFrame]:
    logger.info("Parsing labels for all partitions")
    transformed = [
        LabelTransformer.transform(df, partition)
        for df, partition in zip(dataframes, PARTITIONS)
    ]
    logger.info("Labels parsed for all partitions")
    return transformed


def relocate_documents(dataframes: list[pd.DataFrame]) -> None:
    logger.info("Relocating documents for all partitions")
    DocumentRelocator.relocate(
        dataframes,
        list(PARTITIONS),
        DATA_DIR,
        OUTPUT_DIR,
    )
    logger.info("Documents relocated for all partitions")


def store_parquets(dataframes: list[pd.DataFrame]) -> None:
    for df, partition in zip(dataframes, PARTITIONS):
        logger.info("Storing parquet for partition: %s, shape: %s", partition, df.shape)
        output_dir = OUTPUT_DIR / partition
        output_dir.mkdir(parents=True, exist_ok=True)
        df.to_parquet(output_dir / "data.parquet.gzip", index=False, compression="gzip")
    logger.info("All partitions have been stored as parquet")


def main() -> None:
    logger.info("Starting main pipeline")

    logger.info("Execute data loading")
    train, val, test = load_data()

    logger.info("Execute label parsing")
    train, val, test = parse_labels([train, val, test])

    logger.info("Execute document relocation")
    relocate_documents([train, val, test])

    logger.info("Execute parquet file storage")
    store_parquets([train, val, test])

    logger.info("Pipeline completed")


if __name__ == "__main__":
    main()
