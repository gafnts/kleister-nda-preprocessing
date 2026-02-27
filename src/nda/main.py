from pathlib import Path
from nda.data_loader import DataLoader, Partition


def main() -> None:
    loader = DataLoader(
        data_dir=Path(__file__).parent / "static" / "data",
        output_dir=Path(__file__).parent / "static" / "outputs",
    )

    partitions: tuple[Partition, Partition, Partition] = ("train", "dev-0", "test-A")
    train, val, test = [loader.load(partition) for partition in partitions]

    [
        loader.export(df, partition)
        for df, partition in zip((train, val, test), partitions)
    ]


if __name__ == "__main__":
    main()
