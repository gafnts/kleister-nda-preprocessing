"""
Tests for utils.py
(relocate_documents, to_parquet)
"""

import logging
from pathlib import Path

import pandas as pd
import pytest

from nda.utils import relocate_documents, to_parquet


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    """Create a source directory with a documents folder and two dummy PDFs."""
    docs = tmp_path / "data" / "documents"
    docs.mkdir(parents=True)
    (docs / "alpha.pdf").write_bytes(b"%PDF-alpha")
    (docs / "beta.pdf").write_bytes(b"%PDF-beta")
    return tmp_path / "data"


@pytest.fixture
def output_dir(tmp_path: Path) -> Path:
    return tmp_path / "output"


@pytest.fixture
def single_partition_df() -> pd.DataFrame:
    return pd.DataFrame({"filename": ["alpha.pdf", "beta.pdf"], "other": [1, 2]})


class TestRelocateDocuments:
    def test_copies_files_to_partition_subdirectory(
        self,
        data_dir: Path,
        output_dir: Path,
        single_partition_df: pd.DataFrame,
    ) -> None:
        relocate_documents([single_partition_df], ["train"], data_dir, output_dir)
        assert (output_dir / "train" / "documents" / "alpha.pdf").exists()
        assert (output_dir / "train" / "documents" / "beta.pdf").exists()

    def test_copied_content_matches_source(
        self,
        data_dir: Path,
        output_dir: Path,
        single_partition_df: pd.DataFrame,
    ) -> None:
        relocate_documents([single_partition_df], ["train"], data_dir, output_dir)
        content = (output_dir / "train" / "documents" / "alpha.pdf").read_bytes()
        assert content == b"%PDF-alpha"

    def test_handles_multiple_partitions(
        self, data_dir: Path, output_dir: Path
    ) -> None:
        df_train = pd.DataFrame({"filename": ["alpha.pdf"]})
        df_dev = pd.DataFrame({"filename": ["beta.pdf"]})

        relocate_documents([df_train, df_dev], ["train", "dev-0"], data_dir, output_dir)

        assert (output_dir / "train" / "documents" / "alpha.pdf").exists()
        assert (output_dir / "dev-0" / "documents" / "beta.pdf").exists()

    def test_deduplicates_filenames(
        self,
        data_dir: Path,
        output_dir: Path,
    ) -> None:
        df = pd.DataFrame({"filename": ["alpha.pdf", "alpha.pdf"]})
        relocate_documents([df], ["train"], data_dir, output_dir)
        assert (output_dir / "train" / "documents" / "alpha.pdf").exists()

    def test_warns_on_missing_documents(
        self,
        data_dir: Path,
        output_dir: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        df = pd.DataFrame({"filename": ["alpha.pdf", "ghost.pdf"]})

        with caplog.at_level(logging.WARNING):
            relocate_documents([df], ["train"], data_dir, output_dir)

        assert "ghost.pdf" in caplog.text
        assert "1 of 2" in caplog.text

    def test_missing_document_does_not_block_existing(
        self,
        data_dir: Path,
        output_dir: Path,
    ) -> None:
        df = pd.DataFrame({"filename": ["alpha.pdf", "ghost.pdf"]})
        relocate_documents([df], ["train"], data_dir, output_dir)
        assert (output_dir / "train" / "documents" / "alpha.pdf").exists()

    def test_creates_output_directories(
        self,
        data_dir: Path,
        output_dir: Path,
        single_partition_df: pd.DataFrame,
    ) -> None:
        assert not output_dir.exists()
        relocate_documents([single_partition_df], ["train"], data_dir, output_dir)
        assert (output_dir / "train" / "documents").is_dir()


class TestToParquet:
    def test_creates_parquet_file(self, output_dir: Path) -> None:
        df = pd.DataFrame({"col": [1, 2, 3]})
        to_parquet([df], ["train"], output_dir)
        assert (output_dir / "train" / "data.parquet").exists()

    def test_round_trips_through_read(self, output_dir: Path) -> None:
        df = pd.DataFrame({"col_a": ["x", "y"], "col_b": [10, 20]})
        to_parquet([df], ["train"], output_dir)
        result = pd.read_parquet(output_dir / "train" / "data.parquet")
        pd.testing.assert_frame_equal(result, df)

    def test_handles_multiple_partitions(self, output_dir: Path) -> None:
        df_train = pd.DataFrame({"v": [1]})
        df_dev = pd.DataFrame({"v": [2]})

        to_parquet([df_train, df_dev], ["train", "dev-0"], output_dir)

        assert (output_dir / "train" / "data.parquet").exists()
        assert (output_dir / "dev-0" / "data.parquet").exists()

    def test_creates_output_directories(self, output_dir: Path) -> None:
        assert not output_dir.exists()
        df = pd.DataFrame({"v": [1]})
        to_parquet([df], ["train"], output_dir)
        assert (output_dir / "train").is_dir()

    def test_excludes_dataframe_index(self, output_dir: Path) -> None:
        df = pd.DataFrame({"v": [1, 2]}, index=[10, 20])
        to_parquet([df], ["train"], output_dir)
        result = pd.read_parquet(output_dir / "train" / "data.parquet")
        assert list(result.index) == [0, 1]
