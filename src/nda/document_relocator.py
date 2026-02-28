import logging
import shutil
from pathlib import Path

import pandas as pd

from nda.data_loader import Partition

logger = logging.getLogger(__name__)


class DocumentRelocator:
    @staticmethod
    def relocate(
        dataframes: list[pd.DataFrame],
        partitions: list[Partition],
        data_dir: Path,
        output_dir: Path,
    ) -> None:
        for df, partition in zip(dataframes, partitions):
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
