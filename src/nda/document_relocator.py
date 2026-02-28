import shutil
import pandas as pd

from typing import List
from pathlib import Path

from nda.data_loader import Partition


class DocumentRelocator:
    @staticmethod
    def relocate(
        dataframes: List[pd.DataFrame],
        partitions: List[Partition],
        data_dir: Path,
        output_dir: Path,
    ) -> None:
        for df, partition in zip(dataframes, partitions):
            src_docs = data_dir / "documents"
            dst_docs = output_dir / partition / "documents"
            dst_docs.mkdir(parents=True, exist_ok=True)
            filenames = df["filename"].unique()
            for fname in filenames:
                src_file = src_docs / fname
                dst_file = dst_docs / fname
                if src_file.exists():
                    shutil.copy2(src_file, dst_file)
                else:
                    pass
