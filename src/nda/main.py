from pathlib import Path
from nda.data_loader import DataLoader, Partition


DATA_DIR: Path = Path(__file__).parent / "static" / "data"
OUTPUT_DIR: Path = Path(__file__).parent / "static" / "outputs"
PARTITIONS: tuple[Partition, Partition, Partition] = ("train", "dev-0", "test-A")


def load_data() -> None:
    loader = DataLoader(
        data_dir=DATA_DIR,
        output_dir=OUTPUT_DIR,
    )

    train, val, test = [loader.load(partition) for partition in PARTITIONS]

    for df, partition in zip((train, val, test), PARTITIONS):
        loader.export(df, partition)


def main() -> None:
    load_data()


if __name__ == "__main__":
    main()
