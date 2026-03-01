import logging
import shutil
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from nda.data_loader import Partition

logger = logging.getLogger(__name__)


def to_parquet(
    dataframes: Sequence[pd.DataFrame],
    partitions: Sequence[Partition],
    output_dir: Path,
) -> None:
    for df, partition in zip(dataframes, partitions, strict=True):
        partition_dir = output_dir / partition
        partition_dir.mkdir(parents=True, exist_ok=True)
        df.to_parquet(partition_dir / "data.parquet", index=False, compression="gzip")


def relocate_documents(
    dataframes: Sequence[pd.DataFrame],
    partitions: Sequence[Partition],
    data_dir: Path,
    output_dir: Path,
) -> None:
    for df, partition in zip(dataframes, partitions, strict=True):
        src_docs = data_dir / "documents"
        dst_docs = output_dir / partition / "documents"
        dst_docs.mkdir(parents=True, exist_ok=True)
        filenames = df["filename"].unique()
        missing = []
        for filename in filenames:
            src_file = src_docs / filename
            dst_file = dst_docs / filename
            if src_file.exists():
                shutil.copy2(src_file, dst_file)
            else:
                missing.append(filename)
        if missing:
            logger.warning(
                "Partition '%s': %d of %d documents not found: %s",
                partition,
                len(missing),
                len(filenames),
                missing,
            )
